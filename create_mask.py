import json
import os
from PIL import Image, ImageDraw

def create_mask_from_json(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # JSON 파일에서 이미지 크기 가져오기
    target_size = (data['imageWidth'], data['imageHeight'])
    mask = Image.new('L', target_size, 0)
    draw = ImageDraw.Draw(mask)

    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            polygon = [(int(x), int(y)) for x, y in shape['points']]
            draw.polygon(polygon, outline=255, fill=255)

    mask.save(output_path)
    print(f"Mask saved at: {output_path}")

# # 절대 경로 설정
# base_dir = os.path.dirname(os.path.abspath(__file__))
# json_dir = os.path.join(base_dir, 'label')
# mask_dir = os.path.join(base_dir, 'mask')

# # mask 폴더가 없으면 생성
# os.makedirs(mask_dir, exist_ok=True)

# # 1부터 80까지 JSON 파일에 대해 마스크 생성
# for i in range(1, 81):
#     json_path = os.path.join(json_dir, f'{i}.json')
#     output_path = os.path.join(mask_dir, f'{i}.png')
#     create_mask_from_json(json_path, output_path)

base_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(base_dir, 'label')
mask_dir = os.path.join(base_dir, 'mask')

json_path = os.path.join(json_dir, 'test.json')
output_path = os.path.join(mask_dir, 'maskedtest.png')
create_mask_from_json(json_path, output_path)
