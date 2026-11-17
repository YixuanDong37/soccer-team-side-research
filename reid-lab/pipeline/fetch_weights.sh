#!/usr/bin/env bash
# Download the model weights sn-gamestate loads, into its pretrained_models folder, and print
# their checksums. Sources and expected MD5s are Zenodo's published values (records 12626395,
# 10653453, 10604211); yolo11m.pt is the Ultralytics v8.3.0 release asset.
set -u
cd /c/Research/plugins/sn-gamestate || exit 1
mkdir -p pretrained_models/calibration pretrained_models/reid pretrained_models/yolo

get() {
  local url="$1" out="$2"
  if [ -e "$out" ]; then
    echo "skipped $out (already there or still downloading)"
    return
  fi
  if curl -sSL --retry 5 --retry-delay 5 --max-time 3600 -o "$out.part" "$url"; then
    mv "$out.part" "$out"
    echo "downloaded $out $(stat -c %s "$out") bytes"
  else
    echo "FAILED $out"
  fi
}

get "https://zenodo.org/records/12626395/files/SV_kp?download=1" pretrained_models/calibration/SV_kp &
get "https://zenodo.org/records/12626395/files/SV_lines?download=1" pretrained_models/calibration/SV_lines &
get "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1" pretrained_models/reid/prtreid-soccernet-baseline.pth.tar &
get "https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1" pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth &
get "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt" pretrained_models/yolo/yolo11m.pt &
wait

echo "--- expected MD5: SV_kp 322d4a6c82d2966ea88b69963ba85f07, SV_lines 8bab28ca07ef2d6cefed9deae4be0989,"
echo "    prtreid 9633825232bc89f23a94522c5561650e, hrnet 58ea12b0420aa3adaa2f74114c9f9721"
md5sum pretrained_models/calibration/SV_kp pretrained_models/calibration/SV_lines \
  pretrained_models/reid/prtreid-soccernet-baseline.pth.tar \
  pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth
sha256sum pretrained_models/yolo/yolo11m.pt
