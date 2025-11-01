import os  
import json  
import numpy as np  
from PIL import Image, ImageDraw  
import glob  
from tqdm import tqdm  
import shutil  
from sklearn.model_selection import train_test_split  

def create_binary_mask_from_json(image_path, json_path, output_path):  

    with Image.open(image_path) as img:  
        width, height = img.size  
    mask = Image.new('L', (width, height), 0)  
    draw = ImageDraw.Draw(mask)  
    with open(json_path, 'r', encoding='utf-8') as f:  
        annotation = json.load(f)  
    

    if 'shapes' in annotation:  
        for shape in annotation['shapes']:  
            points = shape['points']  
            polygon = [(int(x), int(y)) for x, y in points]  
            draw.polygon(polygon, fill=255)  
    mask.save(output_path)  
    return mask  

def process_dataset(source_dir, output_dir, train_ratio=0.8):  
 
    os.makedirs(os.path.join(output_dir, 'train', 'imgs'), exist_ok=True)  
    os.makedirs(os.path.join(output_dir, 'train', 'masks'), exist_ok=True)  
    os.makedirs(os.path.join(output_dir, 'val', 'imgs'), exist_ok=True)  
    os.makedirs(os.path.join(output_dir, 'val', 'masks'), exist_ok=True)  
    

    image_files = glob.glob(os.path.join(source_dir, '*.jpg'))  
    image_files.sort()   
    

    if not image_files:  
        raise ValueError(f"错误: 在 {source_dir} 中没有找到任何JPG图片文件")  
    
    print(f"找到 {len(image_files)} 个图片文件")  
    

    train_files, val_files = train_test_split(  
        image_files,  
        train_size=train_ratio,  
        random_state=42  
    )  
    
    print(f"划分结果: 训练集 {len(train_files)} 个文件, 验证集 {len(val_files)} 个文件")  
    

    def process_files(files, subset):  
        processed_count = 0  
        for img_path in tqdm(files, desc=f"处理{subset}集"):  
            try:  
                base_name = os.path.splitext(os.path.basename(img_path))[0]  
                json_path = os.path.join(source_dir, f"{base_name}.json")  
                
                if not os.path.exists(json_path):  
                    print(f"警告: 未找到对应的JSON文件 - {json_path}")  
                    continue  
                
                dst_img = os.path.join(output_dir, subset, 'imgs', f"{base_name}.png")  
                dst_mask = os.path.join(output_dir, subset, 'masks', f"{base_name}_mask.png")  
                 
                with Image.open(img_path) as img:  
                    img.save(dst_img, 'PNG')  
                
                create_binary_mask_from_json(img_path, json_path, dst_mask)  
                
                processed_count += 1  
                
            except Exception as e:  
                print(f"处理文件 {base_name} 时出错: {str(e)}")  
                continue  
                
        return processed_count  
    
    print("\n处理训练集...")  
    train_processed = process_files(train_files, 'train')  
    
    print("\n处理验证集...")  
    val_processed = process_files(val_files, 'val')  
    
    print("\n数据集处理完成!")  
    print(f"训练集: 成功处理 {train_processed}/{len(train_files)} 个文件")  
    print(f"验证集: 成功处理 {val_processed}/{len(val_files)} 个文件")  
    
    return train_processed + val_processed  

def verify_dataset(output_dir):  
    """  
    验证处理后的数据集  
    """  
    print("\n验证数据集...")  
    
    for subset in ['train', 'val']:  
        imgs_dir = os.path.join(output_dir, subset, 'imgs')  
        masks_dir = os.path.join(output_dir, subset, 'masks')  
        
        imgs_count = len(glob.glob(os.path.join(imgs_dir, '*.png')))  
        masks_count = len(glob.glob(os.path.join(masks_dir, '*.png')))  
        
        print(f"\n{subset}集:")  
        print(f"图片数量: {imgs_count}")  
        print(f"掩码数量: {masks_count}")  
        
        if imgs_count != masks_count:  
            print(f"警告: {subset}集的图片和掩码数量不匹配!")  

if __name__ == "__main__":  

    source_dir = r"your address"  
    output_dir = r"your address"  
    print("数据集处理工具")  
    print(f"源目录: {source_dir}")  
    print(f"目标目录: {output_dir}")  
    print("\n将创建的目录结构:")  
    print("data/")  
    print("  train/")  
    print("    imgs/")  
    print("    masks/")  
    print("  val/")  
    print("    imgs/")  
    print("    masks/")  
    
    confirmation = input("\n是否继续？(y/n): ")  
    
    if confirmation.lower() == 'y':  
        try:  
            total_processed = process_dataset(source_dir, output_dir)  
            verify_dataset(output_dir)  
            print(f"\n成功完成! 总共处理了 {total_processed} 个文件")  
        except Exception as e:  
            print(f"\n处理过程中出错: {str(e)}")  
    else:  
        print("操作已取消")
