import cv2

def get_grid_final_shape(num_workers, img_shape=(200, 256), rows=2, cols=2, label_h=30, border=2):
    """
    计算监控网格最终生成的 (宽度, 高度)
    img_shape: (W, H)
    """
    img_h, img_w = img_shape
    
    # 单个单元格的计算
    # 高度 = 图像高 + 标签高 + 上下边框
    cell_h = img_h + label_h + (2 * border)
    # 宽度 = 图像宽 + 左右边框
    cell_w = img_w + (2 * border)
    
    # 最终大图尺寸
    final_h = rows * cell_h
    final_w = cols * cell_w
    
    return (final_w, final_h)

def create_monitor_grid(images, labels=None, rows=2, cols=2):
    """
    极速版监控网格：输入 BGR 图像，输出 RGB 格式网格
    images shape: (NUM_WORKERS, 200, 256, 3) 格式为 BGR
    """
    num_imgs = images.shape[0]
    img_h, img_w = 200, 256 
    label_h = 30
    border = 2
    
    cell_h = img_h + label_h
    cell_w = img_w
    
    # 1. 预分配大画布 (注意：如果你最后显示的地方期望 RGB，这里颜色定义也要按 RGB 来)
    grid_h = rows * (cell_h + 2 * border)
    grid_w = cols * (cell_w + 2 * border)
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # RGB 格式下的颜色定义
    border_color = (0, 255, 0)   # 绿色 (R=0, G=255, B=0)
    label_bg_color = (50, 50, 50) # 深灰
    text_color = (255, 255, 255) # 白色

    for i in range(min(num_imgs, rows * cols)):
        r, c = divmod(i, cols)
        
        y_start = r * (cell_h + 2 * border) + border
        x_start = c * (cell_w + 2 * border) + border
        
        # --- 步骤 A: 填充标签栏 ---
        grid[y_start : y_start + label_h, x_start : x_start + cell_w] = label_bg_color
        
        # --- 步骤 B: 核心转换 (BGR -> RGB) ---
        # 既然输入是 BGR，我们用 [..., ::-1] 把它倒序变成 RGB 填入画布
        img_y_start = y_start + label_h
        grid[img_y_start : img_y_start + img_h, x_start : x_start + img_w] = images[i][..., ::-1]
        
        # --- 步骤 C: 边框和文字 ---
        # 注意：cv2 的绘图函数默认还是按 BGR 处理颜色参数的
        # 如果 grid 最终是给 plt.imshow 或其他 RGB 渲染器看，
        # 我们传入给 cv2.rectangle 的颜色元组也得反着写。
        
        # 绘制边框 (因为 grid 现在是 RGB 序，所以传 (0,255,0) 依然是绿)
        cv2.rectangle(grid, (x_start - border, y_start - border), 
                      (x_start + cell_w + border - 1, y_start + cell_h + border - 1), 
                      border_color, border)
        
        if labels:
            # 写字
            cv2.putText(grid, labels[i], (x_start + 10, y_start + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    return grid
