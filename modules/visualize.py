import os

import torch

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
from pygame.locals import *
import sys
from OpenGL.GL import *
from cuda.bindings import driver as cu
import torch.multiprocessing as mp
import queue
import time
from typing import Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration settings for the CUDA-OpenGL visualization."""
    TARGET_FPS = 10
    MAX_QUEUE_SIZE = 100


# =============================================================================
# CUDA Helper Functions
# =============================================================================

def check_cuda_error(result):
    """Check CUDA error and raise RuntimeError if error occurred."""
    if isinstance(result, tuple):
        err = result[0] if len(result) > 0 else result
    else:
        err = result

    if err != cu.CUresult.CUDA_SUCCESS:
        try:
            error_string = cu.cuGetErrorString(err)
            if isinstance(error_string, tuple):
                error_msg = error_string[1] if len(error_string) > 1 else str(error_string[0])
            else:
                error_msg = str(error_string)
        except:
            error_msg = f"CUDA Error Code: {err}"
        raise RuntimeError(f"CUDA Error: {error_msg}")


def initialize_cuda():
    """Initialize CUDA driver API."""
    result = cu.cuInit(0)
    check_cuda_error(result)

    # Get current device - may not be needed in this context
    try:
        result = cu.cuCtxGetDevice()
        if isinstance(result, tuple) and len(result) > 1:
            err, device = result
            check_cuda_error(err)
            return device
        else:
            # If no current context, just return 0
            return 0
    except:
        # If no current context exists, that's fine
        return 0


# =============================================================================
# Visualization Hook Module
# =============================================================================

class VisualizationHook(torch.nn.Identity):
    """A PyTorch module that visualizes tensors in a separate process."""
    _queue = None
    _process = None
    _lock = mp.Lock()

    def __init__(self, name: str, gamma: float = 0.2):
        super().__init__()
        if not isinstance(name, str) or not name:
            raise ValueError("VisualizationHook must have a non-empty string name.")
        self.name = name

        with VisualizationHook._lock:
            if VisualizationHook._process is None:
                ctx = mp.get_context('spawn')
                VisualizationHook._queue = ctx.Queue(maxsize=Config.MAX_QUEUE_SIZE)
                VisualizationHook._process = ctx.Process(
                    target=start_visualizer,
                    args=(VisualizationHook._queue, gamma)
                )
                VisualizationHook._process.start()
                print("Visualization process started.")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if VisualizationHook._queue:
            try:
                VisualizationHook._queue.put_nowait((self.name, tensor.detach()))
            except queue.Full:
                pass
        return tensor

    @classmethod
    def stop_visualization(cls):
        """Stop the shared visualization system."""
        with cls._lock:
            if cls._process and cls._process.is_alive():
                print("Stopping visualization process...")
                if cls._queue:
                    try:
                        cls._queue.put("STOP")
                    except Exception:
                        pass
                cls._process.join(timeout=5)
                if cls._process.is_alive():
                    cls._process.terminate()
                cls._process = None
                cls._queue = None
                print("Visualization process stopped.")


def start_visualizer(comm_queue, initial_gamma):
    """Target function for the spawned process."""
    TensorVisualizer(comm_queue, initial_gamma).run()


# =============================================================================
# Tensor Visualizer Class (Process Target)
# =============================================================================

class TensorVisualizer:
    """
    Real-time tensor visualizer using CUDA-OpenGL interop.
    This class is designed to run in a separate process.
    """

    def __init__(self, comm_queue: mp.Queue, initial_gamma: float = 0.2):
        self.queue = comm_queue
        self.gamma = initial_gamma
        self.tensors = {}
        self.names = []
        self.active_hook_index = 0

        self.inferno_cmap = torch.tensor([
            [0.001462, 0.000466, 0.013866], [0.042253, 0.028139, 0.141141],
            [0.122908, 0.047536, 0.281624], [0.217949, 0.036615, 0.383522],
            [0.328921, 0.057827, 0.427511], [0.472328, 0.110547, 0.428334],
            [0.621685, 0.164184, 0.388781], [0.735683, 0.215906, 0.330245],
            [0.823386, 0.275197, 0.266085], [0.885302, 0.342586, 0.202968],
            [0.923215, 0.399359, 0.155193], [0.949562, 0.455660, 0.110164],
            [0.974176, 0.536780, 0.048392], [0.985315, 0.608422, 0.024202],
            [0.987819, 0.652773, 0.045581], [0.982228, 0.751442, 0.147565],
            [0.966243, 0.836191, 0.261534], [0.946392, 0.930761, 0.442367],
            [0.957896, 0.971003, 0.556275], [0.988362, 0.998364, 0.644924]
        ], dtype=torch.float32, device='cuda')

        self.clock = None
        self.pbo = None
        self.texture = None
        self.cuda_graphics_resource = None
        self.buffer_size = None
        self.current_display_size = None
        self.current_tensor_shape = None
        self.pygame_initialized = False
        self.cuda_context = None

    def _is_stop_item(self, item):
        return isinstance(item, str) and item == "STOP"

    def run(self):
        """Main visualization loop."""
        opengl_initialized = False
        try:
            # Initialize CUDA in this process
            initialize_cuda()

            while True:
                if opengl_initialized:
                    if not self._handle_events():
                        break

                try:
                    while True:
                        item = self.queue.get_nowait()
                        if self._is_stop_item(item):
                            break
                        name, tensor = item
                        if name not in self.tensors:
                            self.names.append(name)
                        self.tensors[name] = tensor
                except queue.Empty:
                    pass

                active_name = None
                current_tensor = None
                if self.names:
                    if self.active_hook_index >= len(self.names):
                        self.active_hook_index = 0
                    active_name = self.names[self.active_hook_index]
                    current_tensor = self.tensors.get(active_name)

                if current_tensor is not None:
                    if not opengl_initialized or self._needs_resize(current_tensor.shape):
                        self._setup_opengl(current_tensor.shape)
                        opengl_initialized = True

                    try:
                        display_tensor = self._prepare_tensor_for_display(current_tensor, active_name)
                        self._transfer_to_opengl(display_tensor)
                        self._render_frame()
                        self._update_display(active_name)
                    except Exception as e:
                        print(f"Error rendering tensor for hook '{active_name}': {e}")
                        time.sleep(1 / Config.TARGET_FPS)
                else:
                    if opengl_initialized:
                        glClear(GL_COLOR_BUFFER_BIT)
                        self._update_display(active_name)
                    else:
                        time.sleep(1 / Config.TARGET_FPS)

        except Exception as e:
            print(f"Visualization process error: {e}")
        finally:
            if opengl_initialized:
                self._cleanup_resources()

    def _calculate_display_size(self, tensor_shape: Tuple[int, ...]) -> Tuple[int, int]:
        if len(tensor_shape) >= 2:
            height, width = tensor_shape[-2], tensor_shape[-1]
        else:
            height, width = 256, 256
        display_width = (width // 2) * 2
        display_height = (height // 2) * 2
        return (max(2, display_width), max(2, display_height))

    def _setup_opengl(self, tensor_shape: Tuple[int, ...]):
        display_width, display_height = self._calculate_display_size(tensor_shape)
        self.current_display_size = (display_width, display_height)
        self.current_tensor_shape = tensor_shape

        if not self.pygame_initialized:
            pygame.init()
            self.pygame_initialized = True

        pygame.display.set_mode(self.current_display_size, OPENGL | DOUBLEBUF | RESIZABLE)
        pygame.display.set_caption(f"NN Tensor Viz - Shape: {tensor_shape}")
        self.clock = pygame.time.Clock()

        glViewport(0, 0, display_width, display_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, display_width, 0, display_height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        self._cleanup_gl_resources()

        tensor_h, tensor_w = tensor_shape[-2:]
        self.buffer_size = tensor_w * tensor_h * 3
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self.buffer_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tensor_w, tensor_h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Register OpenGL buffer with CUDA using new driver API
        result = cu.cuGraphicsGLRegisterBuffer(
            self.pbo, cu.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
        )
        if isinstance(result, tuple):
            err, self.cuda_graphics_resource = result
            check_cuda_error(err)
        else:
            check_cuda_error(result)
            self.cuda_graphics_resource = result

    def _needs_resize(self, tensor_shape: Tuple[int, ...]) -> bool:
        if self.current_tensor_shape is None:
            return True
        if len(tensor_shape) >= 2 and len(self.current_tensor_shape) >= 2:
            return self.current_tensor_shape[-2:] != tensor_shape[-2:]
        return tensor_shape != self.current_tensor_shape

    def _cleanup_gl_resources(self):
        if self.cuda_graphics_resource:
            try:
                result = cu.cuGraphicsUnregisterResource(self.cuda_graphics_resource)
                check_cuda_error(result)
            except Exception:
                pass
            self.cuda_graphics_resource = None
        if self.pbo:
            try:
                glDeleteBuffers(1, [self.pbo])
            except Exception:
                pass
            self.pbo = None
        if self.texture:
            try:
                glDeleteTextures([self.texture])
            except Exception:
                pass
            self.texture = None

    def _prepare_tensor_for_display(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if not tensor.is_cuda:
            tensor = tensor.cuda()

        if tensor.dim() == 4:
            tensor = tensor[0]
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[0] > 0:
            tensor = tensor[0].unsqueeze(0)
        else:
            h, w = tensor.shape[-2:]
            tensor = torch.zeros(1, h, w, device=tensor.device, dtype=tensor.dtype)

        tensor = torch.clamp(tensor, 0.0, 1.0)
        tensor = torch.pow(tensor, self.gamma)
        indices = (tensor.squeeze(0) * (self.inferno_cmap.shape[0] - 1)).long()
        tensor_hwc = self.inferno_cmap[indices]
        tensor_8bit = (tensor_hwc * 255).byte().contiguous()

        if tensor_8bit.numel() != self.buffer_size:
            raise RuntimeError(f"Tensor size mismatch: expected {self.buffer_size}, got {tensor_8bit.numel()}")
        return tensor_8bit

    def _transfer_to_opengl(self, gpu_image_8bit: torch.Tensor):
        # Map the graphics resource using new driver API
        result = cu.cuGraphicsMapResources(1, self.cuda_graphics_resource, cu.CUstream(0))
        check_cuda_error(result)

        try:
            # Get mapped pointer using new driver API
            result = cu.cuGraphicsResourceGetMappedPointer(self.cuda_graphics_resource)
            if isinstance(result, tuple):
                err, d_ptr, size = result
                check_cuda_error(err)
            else:
                raise RuntimeError("Unexpected return format from cuGraphicsResourceGetMappedPointer")

            if size < gpu_image_8bit.numel():
                raise RuntimeError(f"Mapped PBO size {size} is smaller than required {gpu_image_8bit.numel()}")

            # Copy data using new driver API
            result = cu.cuMemcpyDtoD(d_ptr, gpu_image_8bit.data_ptr(), gpu_image_8bit.numel())
            check_cuda_error(result)

        finally:
            result = cu.cuGraphicsUnmapResources(1, self.cuda_graphics_resource, cu.CUstream(0))
            check_cuda_error(result)

    def _render_frame(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        tensor_h, tensor_w = self.current_tensor_shape[-2:]
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tensor_w, tensor_h, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self._draw_textured_quad()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

    def _draw_textured_quad(self):
        win_w, win_h = self.current_display_size
        tensor_h, tensor_w = self.current_tensor_shape[-2:]

        if win_h <= 0 or tensor_h <= 0:
            return

        win_aspect = win_w / win_h
        tensor_aspect = tensor_w / tensor_h

        if win_aspect > tensor_aspect:
            quad_h = win_h
            quad_w = quad_h * tensor_aspect
            x_offset = (win_w - quad_w) / 2
            y_offset = 0
        else:
            quad_w = win_w
            quad_h = quad_w / tensor_aspect
            x_offset = 0
            y_offset = (win_h - quad_h) / 2

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0);
        glVertex2f(x_offset, y_offset + quad_h)
        glTexCoord2f(1.0, 1.0);
        glVertex2f(x_offset + quad_w, y_offset + quad_h)
        glTexCoord2f(1.0, 0.0);
        glVertex2f(x_offset + quad_w, y_offset)
        glTexCoord2f(0.0, 0.0);
        glVertex2f(x_offset, y_offset)
        glEnd()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return False
            elif event.type == KEYDOWN:
                if event.key == K_UP:
                    self.gamma = min(5.0, self.gamma + 0.1)
                elif event.key == K_DOWN:
                    self.gamma = max(0.1, self.gamma - 0.1)
                elif event.key == K_RIGHT:
                    if self.names: self.active_hook_index = (self.active_hook_index + 1) % len(self.names)
                elif event.key == K_LEFT:
                    if self.names: self.active_hook_index = (self.active_hook_index - 1 + len(
                        self.names)) % len(self.names)
            elif event.type == VIDEORESIZE:
                self.current_display_size = (event.w, event.h)
                pygame.display.set_mode(self.current_display_size, OPENGL | DOUBLEBUF | RESIZABLE)
                glViewport(0, 0, event.w, event.h)
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                glOrtho(0, event.w, 0, event.h, -1, 1);
                glMatrixMode(GL_MODELVIEW);
                glLoadIdentity()
        return True

    def _update_display(self, active_name: Optional[str]):
        pygame.display.flip()
        fps = self.clock.get_fps()
        name_str = f"Hook: {active_name}" if active_name else "Waiting..."
        pygame.display.set_caption(f"{name_str} | FPS: {fps:.1f} | Gamma: {self.gamma:.2f} (Up/Down) | Cycle: (L/R)")
        self.clock.tick(Config.TARGET_FPS)

    def _cleanup_resources(self):
        self._cleanup_gl_resources()
        if self.pygame_initialized:
            pygame.quit()


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example of how to use the VisualizationHook with different tensor sizes."""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(),
        VisualizationHook(name="conv1_output"),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
        VisualizationHook(name="conv2_output"),
        torch.nn.Conv2d(128, 1, 1),
        VisualizationHook(name="single_channel"),
        torch.nn.AdaptiveAvgPool2d((64, 64)),
        VisualizationHook(name="pooled_output"),
        torch.nn.Flatten(),
        torch.nn.Linear(1 * 64 * 64, 10)
    ).cuda()

    test_sizes = [(1, 3, 224, 224), (1, 3, 512, 256), (1, 3, 128, 512)]

    try:
        print("Running forward pass with multiple named hooks...")
        print("Use Left/Right arrow keys in the window to cycle between hooks.")
        for size in test_sizes:
            print(f"\nTesting with input size: {size}")
            test_input = torch.rand(*size).cuda()
            with torch.no_grad():
                for i in range(20):
                    model(test_input)
                    time.sleep(0.1)
                    test_input = (test_input + torch.randn_like(test_input) * 0.05).clamp(0, 1)
            time.sleep(2)
    finally:
        VisualizationHook.stop_visualization()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires CUDA.")
        sys.exit(1)

    # This is crucial for CUDA + multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    print("Starting VisualizationHook example with named hooks...")
    example_usage()
