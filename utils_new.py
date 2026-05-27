import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import re
import cv2
import numpy as np

"""Code is from here: https://github.com/SKaiNET-developers/SKaiNET/issues/262"""
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image to new_shape while retaining aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute unpadded dimensions
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Compute padding (width, height)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    # Divide padding equally on both sides
    dw /= 2
    dh /= 2

    # Resize if necessary
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add border
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img



#The functions in this file contain AI generated code (GOOGLE GEMINI)
class CalibrationDataset(Dataset):
    """
    Loads raw images from a folder — no labels needed.
    Used to run the folded model in train mode so BN running stats update.
    """
    def __init__(self, img_dir, img_size=640):
        self.img_size = img_size
        self.paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.png"))
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in: {img_dir}")
        print(f"   [Calibration] Found {len(self.paths)} images in {img_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        # ToTensor() gives [0,1] float — multiply by 255 so reset_bn_stats
        # can divide by 255 and feed YOLO the expected [0,1] range
        tensor = transforms.ToTensor()(img) * 255.0   # [3, H, W] in [0, 255]
        return tensor, 0   # dummy label — not used


class COCOImageFolder(Dataset):
    """
    Image-only dataset for BN recalibration (REPAIR forward pass).
    Uses letterboxing to match YOLOv8 evaluation preprocessing.
    """

    def __init__(self, image_dir, imgsz=640, max_images=None):
        self.imgsz = imgsz  # <-- CRITICAL: Save this so letterbox can use it

        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(exts)
        ])

        if max_images is not None:
            self.paths = self.paths[:max_images]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Use self.paths to perfectly match the __init__ definition above
        img_path = self.paths[index]
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")

        # 1. Apply Letterbox (using the saved self.imgsz)
        img = letterbox(img, new_shape=(self.imgsz, self.imgsz))

        # 2. Convert BGR to RGB (YOLO format)
        img = img[:, :, ::-1]

        # 3. Convert HWC to CHW
        img = img.transpose((2, 0, 1))

        # 4. Normalize to [0, 1]
        img_tensor = torch.from_numpy(img.copy()).float() / 255.0

        # Return as a tuple so `batch[0]` works in your REPAIR loop
        return img_tensor,

def test_forward_pass(model, device):
    """Runs a single dummy image through the model to check for crashes."""
    print("\nTesting forward pass...")
    try:
        # Standard YOLO input size is 640x640
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print("Forward pass successful! No shape mismatches.")
        # YOLO output is usually a list; index 0 is the raw detections
        print(f"Output shape: {output[0].shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

def get_module_by_name(model, name):
    for part in name.split('.'):
        model = model[int(part)] if part.isdigit() else getattr(model, part)
    return model

def check_layer_shapes(model, folding_plan, shape_snapshot, hide_bn=False, internal_name=True, show_reduction=True):
    """
    The Definitive Architecture Report.
    - internal_name: Toggles the 'Full Path' column.
    - show_reduction: Toggles 'Orig → Curr' vs 'Curr' only.
    - Cyan block scoping & Yellow ConcatBlock math.
    """
    C = {
        'b': '\033[94m', 'cy': '\033[96m', 'g': '\033[92m',
        'y': '\033[93m', 'r': '\033[91m', 'bold': '\033[1m',
        'dim': '\033[2m', 'res': '\033[0m'
    }

    def clean_len(s):
        return len(re.sub(r'\033\[[0-9;]*m', '', s))

    def align(text, width, side='left'):
        length = clean_len(text)
        padding = " " * (width - length)
        return text + padding if side == 'left' else padding + text

    # 1. DYNAMIC COLUMN WIDTHS
    w_name = 22
    w_path = 32 if internal_name else 0
    w_chan = 14 if show_reduction else 6  # Expand for arrows if needed
    w_fold, w_type, w_res = 10, 15, 12

    # 2. CALCULATE TABLE DIMENSIONS
    num_gaps = 6 if internal_name else 5
    # Math: sum of cols + (3 spaces per gap) + 2 (left pipe) + 2 (right pipe)
    inner_width = w_name + w_path + (w_chan * 2) + w_fold + w_type + w_res + (num_gaps * 3) + 4
    content_width = inner_width - 4

    # Border strings
    top = f"╔{'═' * (inner_width - 2)}╗"
    bottom = f"╚{'═' * (inner_width - 2)}╝"

    # Header construction
    h_parts = [align(C['bold'] + 'Layer' + C['res'], w_name)]
    if internal_name: h_parts.append(align(C['bold'] + 'Full Path' + C['res'], w_path))
    h_parts.extend([align('In', w_chan), align('Out', w_chan), align('Folding', w_fold), align('Block Type', w_type),
                    align('Residual', w_res)])
    header = f"║ {' ║ '.join(h_parts)} ║"

    # Separator construction
    s_parts = [f"{'═' * (w_name + 2)}", f"{'═' * (w_chan + 2)}", f"{'═' * (w_chan + 2)}", f"{'═' * (w_fold + 2)}",
               f"{'═' * (w_type + 2)}", f"{'═' * (w_res + 2)}"]
    if internal_name: s_parts.insert(1, f"{'═' * (w_path + 2)}")
    sep = f"╠{'╬'.join(s_parts)}╣"

    print(f"\n{C['bold']}{top}{C['res']}\n{header}\n{C['bold']}{sep}{C['res']}")

    active_block = None

    for layer_name, settings in folding_plan.items():
        if layer_name == "input" or (hide_bn and ".bn" in layer_name):
            continue

        try:
            module = get_module_by_name(model, layer_name)
            parts = layer_name.split('.')
            p_path = ".".join(parts[:2]) if len(parts) > 2 else None
            p_mod = get_module_by_name(model, p_path) if p_path else None
            is_c2f = "C2f" in str(type(p_mod)) if p_mod else False

            # --- CYAN BLOCK START ---
            if is_c2f and p_path != active_block:
                print(f"║ {align(C['cy'] + C['bold'] + 'BLOCK START: C2F' + C['res'], content_width)} ║")
                active_block = p_path
            elif not is_c2f and active_block:
                print(f"║ {align(C['cy'] + C['bold'] + 'BLOCK END' + C['res'], content_width)} ║")
                active_block = None

            # --- CHANNELS & REDUCTION ---
            orig_in, orig_out = shape_snapshot.get(layer_name, (0, 0))
            curr_in = getattr(module, 'in_channels', getattr(module, 'num_features', 0))
            curr_out = getattr(module, 'out_channels', getattr(module, 'num_features', 0))

            def get_dim_str(orig, curr):
                if not show_reduction or orig == curr:
                    return align(str(curr), w_chan)
                return align(f"{C['dim']}{orig} → {C['res']}{curr}", w_chan)

            in_str = get_dim_str(orig_in, curr_in)
            out_str = get_dim_str(orig_out, curr_out)

            # --- ROW BUILDING ---
            indent_level = len(parts) - 2
            indent = "  " * indent_level + ("└─ " if indent_level > 0 else "")
            d_name = align(f"{C['dim'] if is_c2f else ''}{indent}{parts[-1]}{C['res']}", w_name)

            fold_val = "Yes" if settings.get('do_folding') else "No"
            fold_styled = align(f"{C['b'] if fold_val == 'Yes' else ''}{fold_val}{C['res']}", w_fold)

            b_type, res_styled = "Standalone", align(f"{C['dim']}-{C['res']}", w_res)
            if is_c2f:
                if ".m." in layer_name:
                    b_idx = parts[parts.index('m') + 1]
                    b_type = f"C2f (B{int(b_idx) + 1})"
                    b_mod = get_module_by_name(model, ".".join(parts[:parts.index('m') + 2]))
                    res_styled = align(
                        f"{C['g']}Add{C['res']}" if getattr(b_mod, 'add', False) else f"{C['r']}None{C['res']}", w_res)
                else:
                    b_type = "C2f Internal"

            # Final Row Assembly
            row_elements = [d_name]
            if internal_name: row_elements.append(align(f"{C['dim'] if is_c2f else ''}{layer_name}{C['res']}", w_path))
            row_elements.extend([in_str, out_str, fold_styled, align(b_type, w_type), res_styled])
            print(f"║ {' ║ '.join(row_elements)} ║")

            # --- CONCATBLOCK (Exit Trace) ---
            if is_c2f and layer_name.endswith(".cv2.conv") and ".m." not in layer_name:
                unit = p_mod.cv1.conv.out_channels // 2
                trace = f"{'  ' * indent_level}  └─ {C['y']}ConcatBlock: "
                if show_reduction: trace += f"{orig_in} → {curr_in} "
                trace += f"[{unit}(Id) + {unit}(Split)"
                for i in range(len(p_mod.m)): trace += f" + {unit}(B{i + 1})]"
                print(f"║ {align(trace + C['res'], content_width)} ║")

        except:
            continue

    if active_block:
        print(f"║ {align(C['cy'] + C['bold'] + 'BLOCK END' + C['res'], content_width)} ║")
    print(f"{C['bold']}{bottom}{C['res']}")
def count_parameters(model):
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters())