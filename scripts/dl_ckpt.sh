find /workspace/checkpoints/ -type f -name "*.ckpt" > ckpt_files.txt
# adjust as needed:
!scp -v -i /root/.ssh/id_ed25519 -P 22011 root@69.30.85.228:/workspace/checkpoints/QURE_AUGMENTED_2025-05-08-18-20-03/checkpoints/last.ckpt /content/drive/MyDrive/checkpoints/QURE_AUGMENTED/last.ckpt

!mkdir -p /content/drive/MyDrive/checkpoints/QURE_AUGMENTED

# with open('ckpt_files.txt') as f:
#     for line in f:
#         remote_path = line.strip()
#         filename = remote_path.split('/')[-1]
#         local_path = f'/content/drive/MyDrive/checkpoints/QURE_AUGMENTED/{filename}'
#         !scp -v -i /root/.ssh/id_ed25519 -P 22011 root@69.30.85.228:{remote_path} {local_path}
