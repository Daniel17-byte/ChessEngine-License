#!/usr/bin/env python3
"""Small smoke run for the CSV DaniBot trainer."""

from train_csv_danibot import main


if __name__ == "__main__":
    main(
        [
            "--epochs",
            "1",
            "--max-rows",
            "512",
            "--chunk-size",
            "256",
            "--batch-size",
            "64",
            "--checkpoint-every-chunks",
            "2",
            "--progress-every-chunks",
            "1",
            "--model-path",
            "checkpoints/danibot_csv_smoke.pth",
            "--checkpoint-path",
            "checkpoints/danibot_csv_smoke_resume.pt",
            "--no-resume",
        ]
    )
