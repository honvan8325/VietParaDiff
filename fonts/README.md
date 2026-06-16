# Fonts folder

VietParaDiff intentionally loads fonts **only** from this `fonts/` directory or a user supplied
`--font-dir`. It does not scan system font directories.

Place Vietnamese-capable `.ttf` or `.otf` fonts here before running `vpd-synthetic`.
Recommended open font families include Noto Sans, Noto Serif, Roboto, Source Serif, and Literata.

The repository cannot redistribute font binaries here unless you add fonts with licenses that allow
redistribution in your own copy. Use:

```bash
uv run vpd-download-fonts --output fonts
```

on a machine with internet access, or manually copy fonts into this folder.
