#!/usr/bin/env python3
"""
visualize.py - 可视化Laplace方程求解结果
使用matplotlib或ParaView的Python接口
"""

import numpy as np
import sys

def read_vtk(filename):
    """读取VTK格式结果文件"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 查找温度数据
    data_start = -1
    for i, line in enumerate(lines):
        if 'LOOKUP_TABLE default' in line:
            data_start = i + 1
            break
    
    if data_start == -1:
        print("未找到温度数据")
        return None
    
    # 读取温度值
    temps = []
    for line in lines[data_start:]:
        line = line.strip()
        if line:
            try:
                temps.append(float(line))
            except:
                pass
    
    return np.array(temps)

def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "result.vtk"
    temps = read_vtk(filename)
    
    if temps is not None:
        print(f"温度范围: {np.min(temps):.6f} ~ {np.max(temps):.6f}")
        print(f"平均温度: {np.mean(temps):.6f}")
        
        # 简单可视化
        try:
            import matplotlib.pyplot as plt
            
            # 计算网格大小
            size = int(np.sqrt(len(temps)))
            if size * size == len(temps):
                temp_grid = temps.reshape(size, size)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(temp_grid, origin='lower', cmap='hot')
                plt.colorbar(label='Temperature')
                plt.title('Laplace Equation Solution')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.savefig('temperature_distribution.png')
                plt.show()
        except ImportError:
            print("matplotlib未安装，跳过可视化")

if __name__ == "__main__":
    main()