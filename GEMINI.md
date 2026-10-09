## File reading rules

- Do not re-read a file that you have already read in the current session unless:
  1. the file has been modified since it was last read, or
  2. you need a specific section that was not previously available.
- Reuse file contents already present in the conversation/context.
- Before calling read_file or read_many_files, check whether the required content is already in context.
- Avoid repeatedly scanning the project.
