import random
import json

with open("D:\course_work\data\synth_rep\Synthetics\kasthuri_indexed.json", 'r', encoding="utf-8") as f:
    PARAM = json.load(f)

DEBUG_MODE = False

'''
# было первым
PARAM = {

    # Основной цвет фона
    'main_color_mean': 175.0,                   'main_color_std': 15,
    # радиус и сигма размытия конечного изображения
    'main_radius_gausse_blur': 5,              'main_sigma_gausse_blur': 1.95,
    # пределы для разброса нормального шума
    'main_min_gausse_noise_value': 6,          'main_max_gausse_noise_value': 15,
    # цвет включений в фон
    'main_noise_color_mean': 125.0,             'main_noise_color_std': 25,


    # Цвет оболочек АКСОНА
    'axon_shell_color_mean': 65.0,              'axon_shell_color_std': 10.0,
    # Затемнение стандартной текстуры АКСОНА
    'axon_back_color_diff_mean': 50.0,          'axon_back_color_diff_std': 10.0,


    # Толщина линий МЕМБРАН
    'membrane_thickness_mean': 3.0,             'membrane_thickness_std': 1.0,
    # Цвет линий МЕМБРАН
    'membrane_color_mean': 95.0,                'membrane_color_std': 15.0,
    
    
    # Цвет оболочки МИТОХОНДРИИ
    'mitohondrion_shell_color_mean': 85.0,      'mitohondrion_shell_color_std': 5.0,
    # Цвет заполнения МИТОХОНДРИИ
    'mitohondrion_back_color_mean': 120.0,      'mitohondrion_back_color_std': 20.0,
    # Цвет оболочки крист
    'mitohondrion_cristae_shell_color_mean': 80.0, 'mitohondrion_cristae_shell_color_std': 0.0,
    # Цвет внутри крист
    'mitohondrion_cristae_color_mean': 120.0,   'mitohondrion_cristae_color_std': 5.0,
    # толщина оболочки МИТОХОНДРИИ
    'mitohondrion_border_w_min': 3.0,           'mitohondrion_border_w_max': 6.0,


    # Цвет основной части PSD
    'psd_back_color_mean': 60.0,                'psd_back_color_std': 10.0,
    # Цвет метелки PSD
    'psd_addcolor_mean': 95.0,                  'psd_addcolor_std': 20.0,
    # Цвет нитки поверх PSD
    'psd_centerline_color_mean': 115.0,         'psd_centerline_color_std': 15.0,


    # Цвет оболочки ВЕЗИКУЛ
    'vesicles_shell_color_mean': 83.0,          'vesicles_shell_color_std': 10.0,
    # Цвет внутри ВЕЗИКУЛ
    'vesicles_back_color_mean': 125.0,          'vesicles_back_color_std': 10.0,
    
    # интенсивность шума Пирсона
    'poisson_noise': 10,
}
'''

'''
# было вторым
PARAM = {

    # Основной цвет фона
    'main_color_mean': 175.0,                   'main_color_std': 15,
    # радиус и сигма размытия конечного изображения
    'main_radius_gausse_blur': 5,              'main_sigma_gausse_blur': 1.95,
    # пределы для разброса нормального шума
    'main_min_gausse_noise_value': 6,          'main_max_gausse_noise_value': 15,
    # цвет включений в фон
    'main_noise_color_mean': 30.0,             'main_noise_color_std': 15,
    #  максимальная толцина включений фона
    'main_noise_w' : 3,


    # Цвет оболочек АКСОНА
    'axon_shell_color_mean': 65.0,              'axon_shell_color_std': 10.0,
    # Затемнение стандартной текстуры АКСОНА
    'axon_back_color_diff_mean': 50.0,          'axon_back_color_diff_std': 10.0,


    # Толщина линий МЕМБРАН
    'membrane_thickness_mean': 3.0,             'membrane_thickness_std': 1.0,
    # Цвет линий МЕМБРАН
    'membrane_color_mean': 6.0,                  'membrane_color_std': 5.0,
    # Максимум расстояния между МЕМБРАНами
    'membrane_space_thickness_max': 2.0,
    
    
    # Цвет оболочки МИТОХОНДРИИ
    'mitohondrion_shell_color_mean': 40.0,      'mitohondrion_shell_color_std': 5.0,
    # Цвет заполнения МИТОХОНДРИИ
    'mitohondrion_back_color_mean': 100.0,       'mitohondrion_back_color_std': 20.0,
    # Цвет оболочки крист
    'mitohondrion_cristae_shell_color_mean': 30.0, 'mitohondrion_cristae_shell_color_std': 0.0,
    # Цвет внутри крист
    'mitohondrion_cristae_color_mean': 170.0,   'mitohondrion_cristae_color_std': 5.0,
    # толщина оболочки МИТОХОНДРИИ
    'mitohondrion_border_w_min': 2.0,           'mitohondrion_border_w_max': 4.0,
    # длина митохондрии
    "mit_len_min": 26, "mit_len_max": 196,


    # Цвет основной части PSD
    'psd_back_color_mean': 15.0,                'psd_back_color_std': 10.0,
    # Цвет метелки PSD
    'psd_addcolor_mean': 150.0,                  'psd_addcolor_std': 20.0,
    # Цвет нитки поверх PSD
    'psd_centerline_color_mean': 115.0,         'psd_centerline_color_std': 15.0,


    # Цвет оболочки ВЕЗИКУЛ
    'vesicles_shell_color_mean':20.0,          'vesicles_shell_color_std': 10.0,
    # Цвет внутри ВЕЗИКУЛ
    'vesicles_back_color_mean': 130.0,          'vesicles_back_color_std': 10.0,
    # Толщина ВЕЗИКУЛы
    'vesicles_border_w': 1,
    
    # интенсивность шума Пирсона
    'poisson_noise': 40,
    
    'spam_color_mean': 87, 'spam_color_std': 10
}
'''
# для того чтобы создать json файлик нужно раскоментировать нужный PARAM и сохранение (пока что не посмотрел можно ли делать коментарии в json файле
#with open('setting.json', 'w', encoding='utf-8') as f:
#   json.dump(PARAM, f, indent=4)

def uniform_float(mean, std):
    return random.uniform(mean - std, mean + std)


def uniform_int(mean, std):
    return int(random.uniform(mean - std, mean + std) + 0.5)


def normal_randint(mean, std):
    vals = int(random.gauss(mean, std/3)+0.495)
    return 0 if vals < 0 else 255 if vals > 255 else vals



#def normal_random_float(mean, std):
#    return np.random.normal(mean, std)

#def normal_random_int(mean, std):
#    return int(round(np.random.normal(mean, std))
