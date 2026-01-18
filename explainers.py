import torch
import torch.nn.functional as F

class CAM:
    def __init__(self, model, target_layer_name, fc_layer_name):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.fc_layer_name = fc_layer_name
        self.fc_weights = dict(self.model.named_parameters())[self.fc_layer_name + '.weight']
        
        self.feature_maps = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.feature_maps = output.detach()
            
        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        target_layer.register_forward_hook(hook_fn)

    def generate(self, input_tensor, class_idx=None): 
        with torch.no_grad():
            logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        target_weights = self.fc_weights[class_idx].to(self.feature_maps.device)
        batch, C, H, W = self.feature_maps.shape
        
        cam = torch.matmul(target_weights, self.feature_maps.view(C, -1))
        cam = cam.view(H, W)
        cam -= cam.min()
        cam /= (cam.max() + 1e-7)
        return cam.cpu().detach().numpy()

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = dict(self.model.named_modules())[target_layer_name]
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        with torch.set_grad_enabled(True):
            output = self.model(input_tensor)
        
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
            
            score = output[0, class_idx]
            score.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()
        grad_cam = F.relu(grad_cam) 
        grad_cam = grad_cam.cpu().detach().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-6)
         
        return grad_cam 

class CLIPGradCAM(GradCAM):
    def __init__(self, model, target_layer_name):
        super().__init__(model, target_layer_name)
    def generate(self, input_tensor, text_tokens):
        self.model.zero_grad()
        with torch.set_grad_enabled(True):
            logits_per_image, _ = self.model(input_tensor, text_tokens)
            score = logits_per_image[0, 0]
            score.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        grad_cam = F.relu(grad_cam)
        grad_cam = grad_cam.cpu().detach().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-7)
        return grad_cam
    
class MDETRGradCAM(GradCAM):
    def __init__(self, model, target_layer_name):
        super().__init__(model, target_layer_name)
        
    def generate(self, img_tensor, question, ans_type_str, ans_idx):
        self.model.zero_grad()
        with torch.set_grad_enabled(True):
            memory_cache = self.model(img_tensor, [question], encode_and_save=True)
            outputs = self.model(img_tensor, [question], encode_and_save=False, memory_cache=memory_cache) 
            score = outputs[f"pred_answer_{ans_type_str}"][0, ans_idx]
            score.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()
        grad_cam = F.relu(grad_cam).cpu().detach().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-7)
        return grad_cam

class TransformerGradCAM(GradCAM):
    def __init__(self, model, target_layer_name, backbone_name):
        super().__init__(model, target_layer_name)
        self.backbone_layer = dict(self.model.named_modules())[backbone_name]
        self.backbone_layer.register_forward_hook(self._save_features)
        self.features = None
    def _save_features(self, module, input, output):
        self.features = output.detach()
    def generate(self, img_tensor, question, ans_type_str, ans_idx):
        self.model.zero_grad()
        with torch.set_grad_enabled(True):
            memory_cache = self.model(img_tensor, [question], encode_and_save=True)
            outputs = self.model(img_tensor, [question], encode_and_save=False, memory_cache=memory_cache) 
            print(ans_type_str, ans_idx)
            score = outputs[f"pred_answer_{ans_type_str}"][0, 24]
            print(score)
            score.backward()
        has_cls = hasattr(self.model.transformer, 'CLS') and self.model.transformer.CLS is not None
        offset = 1 if has_cls else 0
        h, w = self.features.shape[2:]
        num_img_tokens = h * w
        act_img = self.activations[offset:offset+num_img_tokens,0,:]
        grad_img = self.gradients[offset:offset+num_img_tokens,0,:]
        weights = torch.mean(grad_img, dim=0)
        grad_cam_1d = torch.sum(weights * act_img, dim=-1)
        grad_cam_1d = F.relu(grad_cam_1d)
        grad_cam = grad_cam_1d.view(h, w).cpu().detach().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-7)
        return grad_cam
        