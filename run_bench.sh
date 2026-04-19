#!/bin/bash
set -e
cd /home/user/BenchMARL
export PYTHONUNBUFFERED=1

# 1. ÉP PYTORCH DÙNG 1 LUỒNG (QUAN TRỌNG ĐỂ CỨU CPU KHI CHẠY SONG SONG)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUTPUTS_DIR="/home/user/BenchMARL/outputs"

BASE=(
    "experiment.loggers=[csv]"
    "experiment.render=false"
    "experiment.max_n_frames=2_000_000"

    # 2. ĐƯA TẤT CẢ LÊN CUDA: VMAS chạy trên GPU, ép về CPU sẽ làm nghẽn bus PCIe
    "experiment.sampling_device=cuda"
    "experiment.train_device=cuda"
    "experiment.buffer_device=cuda"
    
    # Giữ nguyên n_envs, nhưng giảm số vòng lặp minibatch để train nhanh hơn
    "experiment.on_policy_n_envs_per_worker=200"
    "experiment.on_policy_collected_frames_per_batch=120000"
    "experiment.on_policy_minibatch_size=4096"
    "experiment.on_policy_n_minibatch_iters=15" # Giảm từ 25 xuống 15


    "experiment.off_policy_n_envs_per_worker=200"
    "experiment.off_policy_collected_frames_per_batch=120000"
    "experiment.off_policy_train_batch_size=4096" 
    "experiment.off_policy_n_optimizer_steps=200"

    "experiment.evaluation_interval=240000"
    
    # TỐI ƯU NASHCONV: Giảm chu kỳ eval và ép số episode xuống 1 (vì đã có 200 envs)
    "+nashconv.enable=true"
    "+nashconv.eval_interval=1" 
    "+nashconv.br_episodes=1"
    "+nashconv.eval_episodes=1"

    # "+nashconv.enable=true"
    # "+nashconv.eval_interval=1"
    # "+nashconv.br_updates=30"
    # "+nashconv.br_episodes=1"
    # "+nashconv.eval_episodes=3"
    # "+nashconv.br_lr=1e-3"
    # "+nashconv.entropy_coef=0.01"
)

zip_results() {
    local algo=$1
    local zip_path="/home/user/results_${algo}.zip"
    
    echo "Đang gom và tái cấu trúc dữ liệu cho $algo..."

    local stage_dir=$(mktemp -d)
    local algo_dir="${stage_dir}/${algo}"
    mkdir -p "$algo_dir"

    cd "$OUTPUTS_DIR"

    while read -r yaml_file; do
        if grep -q "algorithm=${algo}" "$yaml_file"; then
            local seed=$(grep "seed=" "$yaml_file" | tr -dc '0-9')
            local exp_dir=$(dirname $(dirname "$yaml_file"))
            local dest_dir="${algo_dir}/seed_${seed}"
            rm -rf "$dest_dir"
            cp -a "$exp_dir" "$dest_dir"
            
            echo "  + Đã map: $exp_dir ➔ ${algo}/seed_${seed}"
        fi
    done < <(find . -name "overrides.yaml")

    if [ "$(ls -A "$algo_dir")" ]; then
        (cd "$stage_dir" && zip -r -q "$zip_path" "$algo")
        echo "✔ Đã tạo file ZIP cấu trúc chuẩn: results_${algo}.zip"
    else
        echo "⚠️ Cảnh báo: Không tìm thấy dữ liệu nào cho thuật toán $algo"
    fi
    
    rm -rf "$stage_dir"
    cd /home/user/BenchMARL
}

run_algo() {
    local algo=$1
    shift
    local extra=("$@")

    echo ""
    echo "============================================================"
    echo "🚀 BẮT ĐẦU CHẠY: $(echo $algo | tr 'a-z' 'A-Z')"
    echo "============================================================"

    # 4. CHẠY SONG SONG 3 SEED CÙNG LÚC (Tối ưu cho CPU & VRAM 3090)
    MAX_JOBS=3 

    for seed in $(seq 0 9); do
        echo "  ➜ Đang chạy seed ${seed}/9 trong background..."
        
        # Lưu ý dấu '&' ở cuối cùng để chạy ngầm
        python benchmarl/run.py \
            "algorithm=${algo}" \
            "task=vmas/simple_world_comm" \
            "seed=${seed}" \
            "${extra[@]}" \
            "${BASE[@]}" &

        # Đợi nếu số lượng job đạt giới hạn MAX_JOBS
        if (( $(jobs -r -p | wc -l) >= MAX_JOBS )); then
            wait -n
        fi
    done

    # Chờ tất cả các seed còn lại chạy xong
    wait
    echo "  ✔ Hoàn tất toàn bộ 10 seed cho $algo!"
    zip_results "$algo"
}

if [ $# -eq 0 ]; then
    echo "Cách dùng: $0 <algo1> [algo2] ..."
    echo "Ví dụ:    $0 ippo_vi ippo mappo"
    exit 1
fi

for algo in "$@"; do
    case "$algo" in
        ippo_vi|ippo_vi_no_norm|ippo_vi_no_anchor)
            run_algo "$algo" "algorithm.vi_tau=0.05"
            ;;
        *)
            run_algo "$algo"
            ;;
    esac
done