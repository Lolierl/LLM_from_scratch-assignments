import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def load_log(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def plot_loss(data, out_path=None):
    train_loss = data.get("train_loss", [])
    eval_loss = data.get("eval_loss", [])
    steps = data.get("step", [])

    if len(train_loss) != len(steps):
        raise ValueError(
            f"Length mismatch: train_loss={len(train_loss)}, step={len(steps)}"
        )

    plt.figure(figsize=(8, 5))

    # train loss
    plt.plot(steps, train_loss, label="train loss", linewidth=1.5)

    # eval loss：假设按 eval_interval 均匀记录
    if len(eval_loss) > 0:
        # eval step 对齐：取 step 中对应 eval 的位置
        eval_steps = []
        stride = len(train_loss) // len(eval_loss)
        for i in range(len(eval_loss)):
            idx = min((i + 1) * stride - 1, len(steps) - 1)
            eval_steps.append(steps[idx])

        plt.plot(
            eval_steps,
            eval_loss,
            label="eval loss",
            marker="o",
            linestyle="--",
        )

    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training / Evaluation Loss")
    plt.legend()
    plt.grid(True)

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to train_eval_log.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (e.g. loss.png)",
    )
    args = parser.parse_args()

    data = load_log(args.log)
    plot_loss(data, args.out)


if __name__ == "__main__":
    main()
