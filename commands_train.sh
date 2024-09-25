# train on the amedeo_modigliani subset of fscg-8
python src/train.py \
    --output_dir temp \
    --dataset_dir few-shot-creative-generation-8/amedeo_modigliani \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# train on the apple subset of fscg-8
python src/train.py \
    --output_dir train_output/apple \
    --dataset_dir few-shot-creative-generation-8/apple \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# train on the apple subset of fscg-8
python src/train.py \
    --output_dir train_output/apple \
    --dataset_dir few-shot-creative-generation-8/apple \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# train on the burberry subset of fscg-8
python src/train.py \
    --output_dir train_output/burberry \
    --dataset_dir few-shot-creative-generation-8/burberry \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# train on the frank_gehry subset of fscg-8
python src/train.py \
    --output_dir train_output/frank_gehry \
    --dataset_dir few-shot-creative-generation-8/frank_gehry \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# train on the nouns subset of fscg-8
python src/train.py \
    --output_dir train_output/nouns \
    --dataset_dir few-shot-creative-generation-8/nouns \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# train on the onepiece subset of fscg-8
python src/train.py \
    --output_dir train_output/one_piece \
    --dataset_dir few-shot-creative-generation-8/onepiece \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# train on the pokemon subset of fscg-8
python src/train.py \
    --output_dir train_output/pokemon \
    --dataset_dir few-shot-creative-generation-8/pokemon \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32

# train on the rococo subset of fscg-8
python src/train.py \
    --output_dir train_output/rococo \
    --dataset_dir few-shot-creative-generation-8/rococo \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32
