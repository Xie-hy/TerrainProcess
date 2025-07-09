import pdal

def extract(input_path, output_path, class_value):
    # PDAL pipeline，包含读取、过滤、写入操作

    pipeline = pdal.Pipeline()
    pipeline |= pdal.Reader.las(filename=input_path)
    # outlier剔除场景中过高或过低的异常点
    pipeline |= pdal.Filter.outlier(method='statistical',
                                    multiplier=3.0)
    pipeline |= pdal.Filter.range(limits='Classification[3:3]')

    pipeline |= pdal.Writer.las(filename=output_path)

    # 执行PDAL管道
    count = pipeline.execute()



if __name__ == "__main__":
    # 指定输入点云文件路径
    input_path = "C:/Users/xhy/Desktop/PROJECTTEST1/input_1.las"

    # 指定输出点云文件路径
    output_path = "C:/Users/xhy/Desktop/PROJECTTEST1/output_qiaodun.las"

    # 指定要提取的类别
    class_value_to_extract = 3  # 请根据你的数据集中实际的类别值进行调整

    # 提取并导出指定类别的点云
    extract(input_path, output_path, class_value_to_extract)