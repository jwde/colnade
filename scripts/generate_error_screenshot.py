"""Generate an SVG screenshot of ty catching type errors in Colnade schemas."""

import subprocess
import sys

from rich.console import Console
from rich.text import Text


def main():
    # Run ty check and capture output
    result = subprocess.run(
        [sys.executable, "-m", "ty", "check", "scripts/error_showcase.py"],
        capture_output=True,
        text=True,
    )
    raw = result.stderr or result.stdout

    # Create a recording console
    console = Console(record=True, width=90, force_terminal=True)

    # Print a header
    console.print()
    console.print("  $ ty check error_showcase.py", style="bold white")
    console.print()

    # Parse and render the ty output with colors
    for line in raw.splitlines():
        if line.startswith("error["):
            # Error header line - red and bold
            bracket_end = line.index("]") + 1
            tag = line[:bracket_end]
            rest = line[bracket_end:]
            text = Text()
            text.append(tag, style="bold red")
            text.append(rest, style="bold white")
            console.print(text)
        elif line.strip().startswith("-->"):
            # File location - blue
            console.print(Text(line, style="cyan"))
        elif line.strip().startswith("|"):
            parts = line.split("|", 1)
            if len(parts) == 2:
                prefix = parts[0] + "|"
                content = parts[1]
                text = Text()
                text.append(prefix, style="blue")
                if "^^" in content:
                    # Underline markers - red
                    text.append(content, style="red")
                elif "--" in content or "Incompatible" in content or "Declared" in content:
                    # Annotation lines - yellow
                    text.append(content, style="yellow")
                else:
                    # Code lines
                    text.append(content, style="white")
                console.print(text)
        elif line.startswith("info:"):
            console.print(Text(line, style="dim"))
        elif line.startswith("Found"):
            console.print()
            console.print(Text(line, style="bold red"))
        elif line == "":
            console.print()
        else:
            console.print(line)

    console.print()

    # Export as SVG
    svg = console.export_svg(title="ty catching Colnade type errors")

    output_path = "docs/assets/error-showcase.svg"
    with open(output_path, "w") as f:
        f.write(svg)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
