# clang

- [format](https://clang.llvm.org/docs/ClangFormat.html)

## Format

```bash
clang-format -style=Google -dump-config > .clang-format
find . -iregex '.*\.\(h\|c\|cpp\|cu\)$' | xargs clang-format -i
```

