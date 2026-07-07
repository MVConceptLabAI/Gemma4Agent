@echo off
REM AiReceipes per-test wrapper command file.
REM
REM The matrix runner reads this file before every recipe/test.
REM Blank lines and lines starting with REM, @REM, ::, #, or @echo are ignored.
REM If you add one non-comment command below, it is executed once per test before the model server starts.
REM
REM Useful environment variables exposed to the command:
REM   $result      -> full suite result directory containing all logs/results for this run
REM   $test_result -> per-test metrics directory
REM   $recipe_id   -> recipe id, e.g. inference.matrix-12b-regular-off-ctx32k
REM   $recipe_index -> 01, 02, ...
REM   $model       -> model name sent to the OpenAI-compatible endpoint
REM   $base_url    -> endpoint base URL
REM
REM Because Python's shell=True on Windows executes through cmd.exe, use bash -lc when you need Bash syntax.
REM Example GPU snapshot wrapper; uncomment and adapt one line below:
REM bash -lc 'mkdir -p "$result/wrapper-logs" && { date; echo "$recipe_index $recipe_id model=$model base_url=$base_url"; nvidia-smi; } >> "$result/wrapper-logs/${recipe_index}-${recipe_id//[^A-Za-z0-9_.-]/-}.log" 2>&1'
