#!/bin/bash

# Run auto experiment and format output logs (Mac/Linux version)
claude -p "Read program.md carefully. It contains AUTO_MODE instructions. You must NOT ask for confirmation. Execute the full experiment setup and loop immediately. Start now." --dangerously-skip-permissions --output-format stream-json --verbose | python3 -c '
import sys, json

for line in sys.stdin:
    try:
        obj = json.loads(line.strip())
        t = obj.get("type")
        
        if t == "assistant":
            content_list = obj.get("message", {}).get("content", [])
            if not content_list: continue
            content = content_list[0]
            
            if content.get("type") == "text":
                text = content.get("text")
                if text:
                    print(f"\n \033[36mClaude:\033[0m\n\033[37m{text}\033[0m")
            elif content.get("type") == "tool_use":
                name = content.get("name")
                if name == "Bash":
                    print(f"\n \033[33mExecuting command:\033[0m\n   \033[90m{content.get(\"input\", {}).get(\"command\")}\033[0m")
                elif name == "Read":
                    print(f"\n \033[33mReading file:\033[0m\n   \033[90m{content.get(\"input\", {}).get(\"file_path\")}\033[0m")
                elif name in ["Write", "Edit"]:
                    print(f"\n \033[33m{name}ing file:\033[0m\n   \033[90m{content.get(\"input\", {}).get(\"file_path\")}\033[0m")
        
        elif t == "user":
            content_list = obj.get("message", {}).get("content", [])
            if not content_list: continue
            content = content_list[0]
            
            if content.get("type") == "tool_result":
                if content.get("is_error"):
                    print(f"\n \033[31mError:\033[0m\n   \033[31m{content.get(\"content\")}\033[0m")
                else:
                    out = content.get("content", "")
                    if out:
                        if len(out) > 500: out = out[:500] + "..."
                        if out != "./":
                            print(f"\n \033[90mOutput:\033[0m\n   \033[37m{out}\033[0m")
                            
        elif t == "result":
            duration = round(obj.get("duration_ms", 0) / 1000.0, 1)
            cost = obj.get("total_cost_usd", 0)
            turns = obj.get("num_turns", 0)
            
            print("\n\033[32m" + "="*60 + "\033[0m")
            print("\033[32m Experiment completed!\033[0m")
            print("\033[32m" + "="*60 + "\033[0m")
            print(f"  Total duration: {duration} seconds")
            print(f"  Cost: {cost} USD")
            print(f"  Total turns: {turns}")
            print("\033[32m" + "="*60 + "\033[0m")
    except Exception:
        pass
'
