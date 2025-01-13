#!/bin/bash

# 设置路径
IMAGE_PATH="/home/hello/lpf/4DGaussians/data/toy/images"
DATABASE_PATH="/home/hello/lpf/4DGaussians/data/toy/database.db"
OUTPUT_PATH="/home/hello/lpf/4DGaussians/data/toy/sparse"

# 创建数据库
colmap feature_extractor \
    --database_path $DATABASE_PATH \
    --image_path $IMAGE_PATH \
    --ImageReader.single_camera 1

# 特征匹配
colmap exhaustive_matcher \
    --database_path $DATABASE_PATH

# 创建输出目录
mkdir -p $OUTPUT_PATH

# 稀疏重建
colmap mapper \
    --database_path $DATABASE_PATH \
    --image_path $IMAGE_PATH \
    --output_path $OUTPUT_PATH

# 结果转换为TXT格式
colmap model_converter \
    --input_path $OUTPUT_PATH/0 \
    --output_path $OUTPUT_PATH/0 \
    --output_type TXT
