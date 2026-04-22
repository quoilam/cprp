# Run auto experiment and format output logs
claude.cmd -p "Read program.md carefully. It contains AUTO_MODE instructions. You must NOT ask for confirmation. Execute the full experiment setup and loop immediately. Start now." --dangerously-skip-permissions --output-format stream-json --verbose | ForEach-Object {
    try {
        $obj = $_ | ConvertFrom-Json
        
        switch ($obj.type) {
            'assistant' {
                # Display AI text response
                if ($obj.message.content[0].type -eq 'text') {
                    $text = $obj.message.content[0].text
                    if ($text) {
                        Write-Host "`n Claude:" -ForegroundColor Cyan
                        Write-Host $text -ForegroundColor White
                    }
                }
                # Display command being executed
                if ($obj.message.content[0].type -eq 'tool_use') {
                    $tool = $obj.message.content[0]
                    switch ($tool.name) {
                        'Bash' {
                            Write-Host "`n Executing command:" -ForegroundColor Yellow
                            Write-Host "   $($tool.input.command)" -ForegroundColor Gray
                        }
                        'Read' {
                            Write-Host "`n Reading file:" -ForegroundColor Yellow
                            Write-Host "   $($tool.input.file_path)" -ForegroundColor Gray
                        }
                        'Write' {
                            Write-Host "`n Writing file:" -ForegroundColor Yellow
                            Write-Host "   $($tool.input.file_path)" -ForegroundColor Gray
                        }
                        'Edit' {
                            Write-Host "`n Editing file:" -ForegroundColor Yellow
                            Write-Host "   $($tool.input.file_path)" -ForegroundColor Gray
                        }
                    }
                }
            }
            
            'user' {
                # Display command execution result
                if ($obj.message.content[0].type -eq 'tool_result') {
                    if ($obj.message.content[0].is_error) {
                        Write-Host "`n Error:" -ForegroundColor Red
                        Write-Host "   $($obj.message.content[0].content)" -ForegroundColor Red
                    } elseif ($obj.message.content[0].content) {
                        $output = $obj.message.content[0].content
                        if ($output.Length -gt 500) {
                            $output = $output.Substring(0, 500) + "..."
                        }
                        if ($output -and $output -notmatch '^\./$') {
                            Write-Host "`n Output:" -ForegroundColor DarkGray
                            Write-Host "   $output" -ForegroundColor Gray
                        }
                    }
                }
            }
            
            'result' {
                Write-Host "`n" + "="*60 -ForegroundColor Green
                Write-Host " Experiment completed!" -ForegroundColor Green
                Write-Host "="*60 -ForegroundColor Green
                Write-Host "  Total duration: $([math]::Round($obj.duration_ms/1000, 1)) seconds"
                Write-Host "  Cost: $($obj.total_cost_usd) USD"
                Write-Host "  Total turns: $($obj.num_turns)"
                Write-Host "="*60 -ForegroundColor Green
            }
        }
    } catch {
        # Ignore non-JSON lines
    }
}