# clang

## Format

- [format](https://clang.llvm.org/docs/ClangFormat.html)

```bash
clang-format -style=Google -dump-config > .clang-format
find . -iregex '.*\.\(h\|c\|cpp\|cu\)$' | xargs clang-format -i
```

### Makefile

```bash
STYLE ?= Google

clang-format:
	clang-format -style=$(STYLE) -dump-config > .clang-format

format:
	find . -iregex '.*\.\(h\|c\|cpp\|cu\)$$' | xargs clang-format -i
```

#### Format Commands

- Styles: LLVM, GNU, Google, Chromium, Microsoft, Mozilla, WebKit
- default: Google

#### Create a .clang-format

```bash
make clang-format
make clang-format STYLE=LLVM
```

#### Format code

```bash
make format
```

