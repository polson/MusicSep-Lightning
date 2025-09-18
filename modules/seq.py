import sys

from torch import nn


class Seq(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.name = "Seq"
        self.modules_list = nn.ModuleList(modules)

        new_modules = nn.ModuleList()
        for module in self.modules_list:
            new_modules.append(module)
        self.modules_list = new_modules

    def forward(self, x):
        for i, module in enumerate(self.modules_list):
            try:
                x = self.run_module(x, module)
            except Exception as e:
                if not isinstance(e, SeqException):
                    seq_e = SeqException(str(e))
                    seq_e.add_module(i, module)
                    raise seq_e from None
                else:
                    e.add_module(i, module)
                    raise
        return x

    def run_module(self, x, module):
        if isinstance(x, tuple):
            try:
                x = module(*x)
            except TypeError:
                x = module(x[0])
        else:
            x = module(x)
        return x


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
