#运行此脚本，需要将包含视频文件的目录路径作为参数传递给脚本。脚本将自动遍历该目录中的所有视频文件，并计算它们的同步分数 :
#bash calculate_scores_real_videos.sh /path/to/video/files
#将 /path/to/video/files 替换为包含视频文件的目录的路径。脚本将在指定目录中处理每个视频文件，并将每个视频的 Lip Sync Error 分数追加到 all_scores.txt 文件中。

#一个 Shell 脚本，用于计算和评估 Wav2Lip 模型在实际视频数据集上的 Lip Sync Error (LSE)。
#该脚本自动处理一个指定目录中的所有视频文件，并将每个视频的同步分数输出到一个文本文件中
#脚本逻辑
#删除之前的 all_scores.txt 文件（如果存在）。
#使用 ls $1 获取指定目录（通过脚本的第一个参数 $1 传递）中的所有文件名。
#对于目录中的每个文件：
#使用 run_pipeline.py 脚本处理视频文件。这个步骤可能涉及视频的预处理和运行 SyncNet 模型。
#使用 calculate_scores_real_videos.py 脚本计算每个视频的 Lip Sync Error 分数，并将结果追加到 all_scores.txt 文件中。

rm all_scores.txt
yourfilenames=`ls $1`

for eachfile in $yourfilenames
do
   python run_pipeline.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir
   python calculate_scores_real_videos.py --videofile $1/$eachfile --reference wav2lip --data_dir tmp_dir >> all_scores.txt
done
