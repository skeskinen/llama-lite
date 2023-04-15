python3 ../convert.py ./output/pytorch_model.bin
../quantize output/ggml-model-f32.bin output/ggml-model.bin 2
mkdir -p ../models/lite
mv output/ggml-model.bin ../models/lite