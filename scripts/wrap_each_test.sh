#!/usr/bin/env bash
# AiReceipes per-test wrapper command file for Linux hosts.
#
# The matrix runner reads this file before every recipe/test.
# Blank lines and lines starting with REM, @REM, ::, #, or @echo are ignored.
# If you add one non-comment command below, it is executed once per test before the model server starts.
#
# Useful environment variables exposed to the command:
#   $result      -> full suite result directory containing all logs/results for this run
#   $test_result -> per-test metrics directory
#   $recipe_id   -> recipe id, e.g. inference.matrix-12b-regular-off-ctx32k
#   $recipe_index -> 01, 02, ...
#   $model       -> model name sent to the OpenAI-compatible endpoint
#   $base_url    -> endpoint base URL
#
# Example GPU snapshot wrapper; uncomment and adapt one line below:
# mkdir -p "$result/wrapper-logs" && { date; echo "$recipe_index $recipe_id model=$model base_url=$base_url"; nvidia-smi; } >> "$result/wrapper-logs/${recipe_index}-${recipe_id//[^A-Za-z0-9_.-]/-}.log" 2>&1
