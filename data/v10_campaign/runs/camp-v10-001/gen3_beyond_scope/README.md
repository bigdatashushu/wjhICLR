# gen3_beyond_scope

这一目录是**超出 v10 首期范围（两代）**的第三次运行的残留收据：CLI 曾以
`--max-generations 3` 启动一次 resume，第三代完成了 learning 运行与经验包，
在候选生成时 DeepSeek 调用超时（`OfflineServiceUnavailable`，已修：现在会记
`offline_failure.json` + `blocked` 而不是崩）。

保留原因：它是真实运行的记录，删掉等于抹掉事实。**它不参与两代结论**：
`campaign.json` 的 `generation_receipt_refs` 只列 `gen1` 与 `gen2`。
