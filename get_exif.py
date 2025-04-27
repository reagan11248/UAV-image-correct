
import exifread
from PIL import Image

b = b"\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e"
a = b"\x3c\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20"

aa = ["\x3c\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20"]
bb = ["\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e"]


def get_pos(path):
    '''
    :param path:
    :return:
    dict {
    'GpsLatitude',
    'GpsLongitude',
    'RelativeAltitude',
    'GimbalRollDegree',
    'GimbalYawDegree',
    'GimbalPitchDegree',
    'FlightRollDegree',
    'FlightYawDegree',
    'FlightPitchDegree'
    'RtkStdLon',
    'RtkStdLat',
    'RtkStdHgt',
    'FocalLength'
    }

    '''

    data_dict = {}

    image = Image.open(path)
    # 获取图像的宽度和高度
    width, height = image.size
    data_dict['ImageWidth'] = width
    data_dict['ImageHeight'] = height
    # 关闭图像文件
    image.close()

    # rb是读取二进制文件
    img = open(path, 'rb')
    tags = exifread.process_file(img)

    ## 焦距
    focal = tags['EXIF FocalLength'].values[0]
    data_dict['FocalLength'] = focal.num / focal.den

    lat = tags['GPS GPSLatitude'].values
    data_dict['GpsLatitude'] = lat[0] + lat[1]/60 + lat[2].num / lat[2].den / 3600
    if tags['GPS GPSLatitudeRef'].values == 'S':
        data_dict['GpsLatitude'] = - data_dict['GpsLatitude']

    lon = tags['GPS GPSLongitude'].values
    data_dict['GpsLongitude'] = lon[0] + lon[1] / 60 + lon[2].num / lon[2].den / 3600
    if tags['GPS GPSLongitudeRef'].values == 'W':
        data_dict['GpsLongitude'] = - data_dict['GpsLongitude']

    alt = tags['GPS GPSAltitude'].values
    data_dict['GpsAltitude'] = alt[0].num / alt[0].den


    #eq_focal = tags['EXIF FocalLengthIn35mmFilm'].values[0]
    #data_dict['Eq FocalLength'] = eq_focal

    # 'GPS GPSLatitude' 'GPS GPSLongitude' 'EXIF FocalLength'

    data = bytearray()
    # 标识符
    flag = False

    for i in img.readlines():
        # 按行读取二进制信息，标签成对出现
        if a in i:
            flag = True
        if flag:
            # 把第i行数据复制到新数组中
            data += i
        if b in i:
            break

    keys = ['RelativeAltitude', 'GimbalRollDegree', 'GimbalYawDegree', 'GimbalPitchDegree']

    if len(data) > 0:
        data = str(data.decode('ascii'))
        lines = list(filter(lambda x: 'drone-dji:' in x, data.split()))

        for d in lines:
            d = d.strip().split(":")[1]
            k, v = d.split("=")
            if k in keys:
                v = float(v.replace('"', ''))
                data_dict[k] = v

    return data_dict


if __name__ == '__main__':

    pos = get_pos(r'D:\path\image.JPG')
    print(pos)










