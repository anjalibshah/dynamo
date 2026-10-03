## Working style

Start every task by launching 2 or 3 Task subagents in parallel, in a single
message, each investigating a different part of the problem (for example: one
locates and reads the code most likely responsible, one finds the existing
tests covering it, one searches for related usages or similar past fixes).
Wait for all of their findings, then decide on and make the change yourself.
Later in the task, use parallel subagents again whenever there are two or more
independent things to check.
