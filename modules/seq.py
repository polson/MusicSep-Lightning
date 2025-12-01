import sys
from torch import nn


class Select:
    """Marker to select which kwarg becomes the main input."""

    def __init__(self, name):
        self.name = name


class Seq(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.name = "Seq"

        # Filter out Select markers and track them
        self.select_indices = {}  # index -> kwarg name
        actual_modules = []

        for i, m in enumerate(modules):
            if isinstance(m, Select):
                # This Select applies to the next module
                self.select_indices[len(actual_modules)] = m.name
            else:
                actual_modules.append(m)

        self.modules_list = nn.ModuleList(actual_modules)

    def forward(self, x, **kwargs):
        pending_kwargs = kwargs.copy()

        for i, module in enumerate(self.modules_list):
            # Check if this module should operate on a different input
            if i in self.select_indices:
                kwarg_name = self.select_indices[i]
                if kwarg_name in pending_kwargs:
                    # Swap: current x goes into kwargs, selected kwarg becomes x
                    old_x = x
                    x = pending_kwargs[kwarg_name]
                    pending_kwargs[kwarg_name] = old_x
                else:
                    raise ValueError(f"Select('{kwarg_name}') but '{kwarg_name}' not in kwargs")

            try:
                x = self._run_module(x, module, pending_kwargs)
            except Exception as e:
                if not isinstance(e, SeqException):
                    seq_e = SeqException(str(e))
                    seq_e.add_module(i, module)
                    raise seq_e from None
                else:
                    e.add_module(i, module)
                    raise
        return x

    def _run_module(self, x, module, pending_kwargs):
        if not pending_kwargs:
            result = module(x)
        else:
            import inspect
            sig = inspect.signature(module.forward)
            params = list(sig.parameters.values())

            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

            if has_var_keyword:
                result = module(x, **pending_kwargs)
            else:
                extra_params = [p for p in params[1:] if p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY
                )]

                if not extra_params:
                    result = module(x)
                else:
                    matched_kwargs = {}
                    for param in extra_params:
                        if param.name in pending_kwargs:
                            matched_kwargs[param.name] = pending_kwargs[param.name]

                    if matched_kwargs:
                        result = module(x, **matched_kwargs)
                    else:
                        extra_param_names = [p.name for p in extra_params]
                        pending_keys = list(pending_kwargs.keys())
                        pending_values = list(pending_kwargs.values())
                        args_to_pass = pending_values[:len(extra_params)]

                        if args_to_pass:
                            print(f"Warning: No name match for {module.__class__.__name__}. "
                                  f"Passing kwargs {pending_keys[:len(args_to_pass)]} "
                                  f"positionally to params {extra_param_names[:len(args_to_pass)]}")

                        result = module(x, *args_to_pass)

        # Handle tuple returns: (output, dict_to_merge)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            x, new_kwargs = result
            pending_kwargs.update(new_kwargs)
            return x

        return result

    def __getitem__(self, idx):
        return self.modules_list[idx]

    def __len__(self):
        return len(self.modules_list)

    def __iter__(self):
        return iter(self.modules_list)


class SeqException(Exception):
    def __init__(self, message):
        super().__init__("")
        self.message = message
        self.module = None
        self.module_highlights = []

    def add_module(self, i, module):
        self.module = module
        name = f"({i}): {module.__str__().partition('\n')[0]}"
        self.module_highlights.insert(0, name)

    def clean_module_string(self, module_str):
        lines = module_str.split('\n')
        cleaned_lines = []
        skip_next_paren = False

        for line in lines:
            if "(modules_list): ModuleList(" in line:
                skip_next_paren = True
                continue

            if skip_next_paren and line.strip() == ")":
                skip_next_paren = False
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def __str__(self):
        RED = '\033[91m'
        RESET = '\033[0m'

        modules_str = f"(0): {str(self.module)}"
        modules_str = self.clean_module_string(modules_str)

        deepest_red_module = self.module_highlights[-1] if self.module_highlights else ""
        deepest_red_pos = modules_str.find(deepest_red_module)

        if deepest_red_pos != -1:
            end_of_line = modules_str.find('\n', deepest_red_pos + len(deepest_red_module))
            if end_of_line == -1:
                end_of_line = len(modules_str)

            modules_str = modules_str[:end_of_line]

        colored_modules_str = modules_str
        last_highlighted_pos = 0
        for module_name in self.module_highlights:
            pos = colored_modules_str.find(module_name, last_highlighted_pos)
            if pos != -1:
                before_highlight = colored_modules_str[:pos]
                after_highlight = colored_modules_str[pos + len(module_name):]

                colored_modules_str = before_highlight + f'{RED}{module_name}{RESET}' + after_highlight

                last_highlighted_pos = pos + len(f'{RED}{module_name}{RESET}')

        result = f"\n{colored_modules_str}"
        result += f"\n\nError in {self.module_highlights[0]}: {self.message}"

        return result


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if isinstance(exc_value, SeqException):
        print(exc_value, file=sys.stderr)
    else:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = custom_excepthook
