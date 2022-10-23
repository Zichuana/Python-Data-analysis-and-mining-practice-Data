import matplotlib.pyplot as plt
from mplfonts import use_font

FONT_NAMES = {
    'Noto Sans Mono CJK SC': 'Noto等宽',
    'Noto Serif CJK SC': 'Noto宋体',
    'Noto Sans CJK SC': 'Noto黑体',
    'Source Han Serif SC': '思源宋体',
    'Source Han Mono SC': '思源等宽',
    'SimHei': '微软雅黑'
}


def test_chinese():
    for font_name, desc in FONT_NAMES.items():
        use_font(font_name)
        fig = plt.figure(figsize=(4, 1))

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.text(.1, .6, font_name, fontsize=20)
        plt.text(.1, .2, desc, fontsize=20)

        plt.show()


test_chinese()
