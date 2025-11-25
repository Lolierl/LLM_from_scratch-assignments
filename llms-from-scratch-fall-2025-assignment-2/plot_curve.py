import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_log_data(log_path, context_length, global_batch_size):
    """
    加载日志数据并计算训练的token数量
    
    Args:
        log_path: JSON日志文件路径
        context_length: 上下文长度（每个样本的token数）
        global_batch_size: 全局batch大小
    
    Returns:
        dict: 包含训练和验证数据的字典
    """
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    train_loss = log_data.get("train_loss", [])
    eval_loss = log_data.get("eval_loss", [])
    train_steps = log_data.get("step", [])
    
    if not train_loss or not train_steps:
        raise ValueError(f"日志文件 {log_path} 中没有训练数据")
    
    # 计算每个step对应的token数量
    # tokens = step * global_batch_size * context_length
    train_tokens = [step * global_batch_size * context_length for step in train_steps]
    
    # 为验证损失计算对应的token数
    # 假设验证在特定step进行，需要从训练步数中推断
    eval_tokens = []
    if eval_loss:
        # 根据验证损失数量和训练步数，推断验证点
        if len(eval_loss) > 0:
            # 假设验证是在固定间隔进行的
            total_steps = len(train_steps)
            eval_interval = total_steps // len(eval_loss) if len(eval_loss) > 0 else 1
            
            for i in range(len(eval_loss)):
                # 估计验证发生的步数
                eval_step = train_steps[min((i + 1) * eval_interval - 1, len(train_steps) - 1)]
                eval_tokens.append(eval_step * global_batch_size * context_length)
    
    return {
        'train_tokens': train_tokens,
        'train_loss': train_loss,
        'eval_tokens': eval_tokens,
        'eval_loss': eval_loss,
        'name': Path(log_path).parent.name  # 使用父目录名作为实验名
    }


def plot_multiple_experiments(experiments_data, output_path=None, figsize=(14, 5)):
    """
    绘制多个实验的训练和验证损失曲线
    
    Args:
        experiments_data: 实验数据列表，每个元素是load_log_data返回的字典
        output_path: 图片保存路径（可选）
        figsize: 图片尺寸
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 颜色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    
    # ========== 左图：训练损失 ==========
    for idx, exp_data in enumerate(experiments_data):
        color = colors[idx % len(colors)]
        label = exp_data.get('label', f"Experiment {idx+1}")
        
        ax1.plot(exp_data['train_tokens'], exp_data['train_loss'], 
                color=color, linewidth=1.5, alpha=0.8, label=label)
    
    ax1.set_xlabel('Tokens Trained', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss vs Tokens', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10)
    
    # 使用科学计数法显示token数
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # ========== 右图：验证损失 ==========
    for idx, exp_data in enumerate(experiments_data):
        color = colors[idx % len(colors)]
        label = exp_data.get('label', f"Experiment {idx+1}")
        
        if exp_data['eval_loss']:
            ax2.plot(exp_data['eval_tokens'], exp_data['eval_loss'], 
                    color=color, linewidth=2, marker='o', markersize=5, 
                    alpha=0.8, label=label)
    
    ax2.set_xlabel('Tokens Trained', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss vs Tokens', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10)
    
    # 使用科学计数法显示token数
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n========== 实验统计 ==========")
    for idx, exp_data in enumerate(experiments_data):
        label = exp_data.get('label', f"Experiment {idx+1}")
        print(f"\n【{label}】")
        print(f"总训练tokens: {exp_data['train_tokens'][-1]:,.0f}")
        print(f"初始训练损失: {exp_data['train_loss'][0]:.6f}")
        print(f"最终训练损失: {exp_data['train_loss'][-1]:.6f}")
        print(f"训练损失下降: {exp_data['train_loss'][0] - exp_data['train_loss'][-1]:.6f}")
        
        if exp_data['eval_loss']:
            print(f"验证次数: {len(exp_data['eval_loss'])}")
            print(f"初始验证损失: {exp_data['eval_loss'][0]:.6f}")
            print(f"最终验证损失: {exp_data['eval_loss'][-1]:.6f}")
            min_eval_idx = np.argmin(exp_data['eval_loss'])
            print(f"最佳验证损失: {exp_data['eval_loss'][min_eval_idx]:.6f} " 
                  f"(tokens: {exp_data['eval_tokens'][min_eval_idx]:,.0f})")


def main():
    parser = argparse.ArgumentParser(
        description="绘制训练和验证损失曲线（以训练的token数为x轴）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 单个实验
  python plot_curve.py --log_path log1.json --context_length 1024 --global_batch_size 512 --label "Exp1"
  
  # 两个实验对比
  python plot_curve.py --log_path /thullms/3022377347/checkpoints-4-5-512/train_eval_log.json /thullms/3022377347/checkpoints-4-6-postnorm/train_eval_log.json /thullms/3022377347/checkpoints-4-6-postnormtuned/train_eval_log.json\\
                      --context_length 1024 1024 1024\\
                      --global_batch_size 512 512 512\\
                      --label "prenorm" "postnorm" "post-norm-tuned\\
                      --output comparison_4-6-1.png \\
        """
    )
    
    parser.add_argument("--log_path", type=str, nargs='+', required=True,
                      help="训练日志JSON文件路径（可以指定多个）")
    parser.add_argument("--context_length", type=int, nargs='+', required=True,
                      help="上下文长度（对应每个log_path）")
    parser.add_argument("--global_batch_size", type=int, nargs='+', required=True,
                      help="全局batch大小（对应每个log_path）")
    parser.add_argument("--label", type=str, nargs='+', default=None,
                      help="实验标签（对应每个log_path，可选）")
    parser.add_argument("--output", type=str, default=None,
                      help="输出图片路径（可选，如：loss_curves.png）")
    parser.add_argument("--figsize", type=int, nargs=2, default=[14, 5],
                      help="图片尺寸，格式：宽 高")
    
    args = parser.parse_args()
    
    # 验证参数数量一致性
    n_experiments = len(args.log_path)
    if len(args.context_length) == 1:
        args.context_length = args.context_length * n_experiments
    if len(args.global_batch_size) == 1:
        args.global_batch_size = args.global_batch_size * n_experiments
        
    if len(args.context_length) != n_experiments or len(args.global_batch_size) != n_experiments:
        parser.error("context_length和global_batch_size的数量必须与log_path一致（或为1）")
    
    if args.label and len(args.label) != n_experiments:
        parser.error("label数量必须与log_path一致")
    
    # 加载所有实验数据
    experiments_data = []
    for idx, log_path in enumerate(args.log_path):
        log_path = Path(log_path)
        if not log_path.exists():
            print(f"警告：找不到日志文件 {log_path}，跳过")
            continue
        
        try:
            exp_data = load_log_data(
                log_path=log_path,
                context_length=args.context_length[idx],
                global_batch_size=args.global_batch_size[idx]
            )
            
            # 添加标签
            if args.label:
                exp_data['label'] = args.label[idx]
            else:
                exp_data['label'] = f"Exp {idx+1} (BS={args.global_batch_size[idx]})"
            
            experiments_data.append(exp_data)
            print(f"成功加载: {log_path}")
            
        except Exception as e:
            print(f"错误：无法加载 {log_path}: {e}")
            continue
    
    if not experiments_data:
        print("错误：没有成功加载任何实验数据")
        return
    
    # 绘制损失曲线
    plot_multiple_experiments(
        experiments_data=experiments_data,
        output_path=args.output,
        figsize=tuple(args.figsize)
    )


if __name__ == "__main__":
    main()