import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter.simpledialog as tksd
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
# ------------------ matplotlib 中文 ------------------
plt.rcParams['font.sans-serif'] = ['SimHei']      # Windows 黑体
plt.rcParams['axes.unicode_minus'] = False

# ------------------ 1. 读取 Excel ------------------
def read_airports(excel_path):
    df = pd.read_excel(excel_path, header=None)
    data = df.iloc[10:, :].copy()
    data.columns = [
        '机场名字', '所属省份', '机场类别', '类别级别', '飞行区指标',
        '东经-度', '东经-分', '东经-秒', '北纬-度', '北纬-分', '北纬-秒', '机场标高[m]',
        '端识别号-小', '端识别号-大', '长度[m]', '宽度[m]', '坡度[%]', '表面类型',
        '承载强度[t]', 'PCN值', '滑行道-宽度[m]', '滑行道-表面类型', '滑行道-PCN值',
        '直升机-半径[m]', '直升机-表面类型', '直升机-承载强度[t]', '状态', '备注'
    ]
    lon_lat_cols = ['东经-度', '东经-分', '东经-秒', '北纬-度', '北纬-分', '北纬-秒']
    for col in lon_lat_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['lon'] = data['东经-度'] + data['东经-分']/60 + data['东经-秒']/3600
    data['lat'] = data['北纬-度'] + data['北纬-分']/60 + data['北纬-秒']/3600
    return data[['机场名字', 'lon', 'lat', '所属省份']].dropna()

# ------------------ 2. 距离计算 ------------------
EARTH_R = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))

# ------------------ 3. 自补全输入框 ------------------
class AutocompleteEntry(tk.Entry):
    def __init__(self, master, complete_values, **kwargs):
        super().__init__(master, **kwargs)
        self.complete_values = complete_values
        self.var = tk.StringVar()
        self.configure(textvariable=self.var)
        self.var.trace_add("write", self.on_change)
        self.lb = None
        self.lb_selection = -1

    def on_change(self, *_):
        text = self.var.get()
        if self.lb:
            self.lb.destroy()
            self.lb = None
        if not text:
            return
        matches = [v for v in self.complete_values if text.lower() in v.lower()]
        if not matches:
            return
        h = min(8, len(matches))
        self.lb = tk.Listbox(width=self.cget("width"), height=h)
        self.lb.bind("<ButtonRelease-1>", self.select_click)
        self.bind("<FocusOut>", lambda e: self.lb.destroy() if self.lb else None)
        self.bind("<Up>", self.key_up)
        self.bind("<Down>", self.key_down)
        self.bind("<Return>", self.key_return)
        x, y, _, hei = self.bbox("insert")
        self.lb.place(x=x, y=y+hei+2, anchor="nw")
        for m in matches:
            self.lb.insert(tk.END, m)

    def select_click(self, *_):
        if self.lb and self.lb.curselection():
            self.var.set(self.lb.get(self.lb.curselection()[0]))
            self.lb.destroy()
            self.lb = None

    def key_up(self, _):
        if self.lb and self.lb.size():
            self.lb_selection = max(self.lb_selection - 1, 0)
            self.lb.selection_clear(0, tk.END)
            self.lb.selection_set(self.lb_selection)
            self.lb.activate(self.lb_selection)

    def key_down(self, _):
        if self.lb and self.lb.size():
            self.lb_selection = min(self.lb_selection + 1, self.lb.size()-1)
            self.lb.selection_clear(0, tk.END)
            self.lb.selection_set(self.lb_selection)
            self.lb.activate(self.lb_selection)

    def key_return(self, _):
        if self.lb and self.lb_selection >= 0:
            self.var.set(self.lb.get(self.lb_selection))
            self.lb.destroy()
            self.lb = None
        return "break"

# ------------------ 4.GUI ------------------
class AirportApp:
    def __init__(self, root, df):
        self.root = root
        self.df = df
        self.names = df['机场名字'].tolist()
        self.lons  = df['lon'].tolist()
        self.lats  = df['lat'].tolist()

        root.title("机场信息检索器 v4")
        root.geometry("950x700")
        self.GUI()
        self.reset()

    def GUI(self):
        frm = ttk.Frame(self.root, padding=5)
        frm.pack(fill='x')

        ttk.Label(frm, text="起点机场A:").grid(row=0, column=0, sticky='e')
        self.ent_a = AutocompleteEntry(frm, self.names, width=20)
        self.ent_a.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(frm, text="终点机场B:").grid(row=0, column=2, sticky='e')
        self.ent_b = AutocompleteEntry(frm, self.names, width=20)
        self.ent_b.grid(row=0, column=3, padx=2, pady=2)

        ttk.Label(frm, text="省份:").grid(row=0, column=4, sticky='e')
        self.cmb_prov = ttk.Combobox(frm, width=12, state="readonly")
        self.cmb_prov['values'] = ["全部"] + sorted(self.df['所属省份'].unique().tolist())
        self.cmb_prov.current(0)
        self.cmb_prov.grid(row=0, column=5, padx=2, pady=2)

        ttk.Label(frm, text="距离(km)范围:").grid(row=1, column=0, sticky='e')
        self.min_dist = ttk.Entry(frm, width=8); self.min_dist.insert(0, "0")
        self.min_dist.grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(frm, text="—").grid(row=1, column=2)
        self.max_dist = ttk.Entry(frm, width=8); self.max_dist.insert(0, "99999")
        self.max_dist.grid(row=1, column=3, padx=2, pady=2)

        ttk.Button(frm, text="检索", command=lambda: search(self)).grid(row=1, column=4, padx=5)
        ttk.Button(frm, text="重置", command=self.reset).grid(row=1, column=5, padx=5)

        ttk.Label(frm, text="飞机航程(km):").grid(row=2, column=0, sticky='e')
        self.ent_range = ttk.Entry(frm, width=8); self.ent_range.insert(0, "2000")
        self.ent_range.grid(row=2, column=1, padx=2, pady=2)
        ttk.Button(frm, text="统计中转机场", command=lambda: count_transfer(self)).grid(row=2, column=2, padx=5)

        self.listbox = tk.Listbox(self.root, font=("Consolas", 10))
        self.listbox.pack(fill='both', expand=True, padx=5, pady=5)

    def reset(self):
        self.ent_a.var.set('')
        self.ent_b.var.set('')
        self.min_dist.delete(0, tk.END); self.min_dist.insert(0, "0")
        self.max_dist.delete(0, tk.END); self.max_dist.insert(0, "99999")
        self.cmb_prov.current(0)
        self.ent_range.delete(0, tk.END); self.ent_range.insert(0, "2000")
        self.listbox.delete(0, tk.END)
        for _, row in self.df.iterrows():
            self.listbox.insert(tk.END, f"{row['机场名字']}  经度:{row['lon']:.6f}  纬度:{row['lat']:.6f}")

# ------------------ 5. 类外函数 ------------------
def search(ctx):
    ctx.listbox.delete(0, tk.END)
    try:
        min_d = float(ctx.min_dist.get())
        max_d = float(ctx.max_dist.get())
    except ValueError:
        messagebox.showerror("输入错误", "距离范围请输入数字")
        return

    name_a = ctx.ent_a.var.get().strip()
    name_b = ctx.ent_b.var.get().strip()

    def idx(name):
        return ctx.names.index(name) if name in ctx.names else None

    idx_a, idx_b = idx(name_a), idx(name_b)


    if idx_a is None and idx_b is None:
        plot_distance_histogram(ctx, min_d, max_d)
        return

    results = []

    for i in range(len(ctx.names)):
        for j in range(i + 1, len(ctx.names)):
            if idx_a is not None and i != idx_a: continue
            if idx_b is not None and j != idx_b: continue
            if idx_a is not None and idx_b is not None and (i != idx_a or j != idx_b): continue
            d = haversine(ctx.lats[i], ctx.lons[i], ctx.lats[j], ctx.lons[j])
            if min_d <= d <= max_d:
                results.append((ctx.names[i], ctx.names[j], d))

    if not results:
        ctx.listbox.insert(tk.END, "无符合条件的机场对/航线")
    else:
        for a, b, d in sorted(results, key=lambda x: x[2]):
            ctx.listbox.insert(tk.END, f"{a} → {b}  距离 {d:.2f} km")

def plot_distance_histogram(ctx, min_d, max_d):
    selected_prov = ctx.cmb_prov.get()
    is_all_prov = (selected_prov == "全部")
    choices = (
        ["1. 全部航线", "2. 仅省内航线", "3. 省内+邻省航线"] if is_all_prov
        else ["2. 仅省内航线", "3. 省内+邻省航线"]
    )
    hint = "请输入 1/2/3：" if is_all_prov else "请输入 2/3："
    scene = tksd.askinteger("场景选择", "\n".join(choices) + "\n\n" + hint,
                            minvalue=1 if is_all_prov else 2, maxvalue=3)
    if scene is None:
        return

    province_map = dict(zip(ctx.df['机场名字'], ctx.df['所属省份']))
    neighbor_map = {
        '北京': ['天津', '河北'], '天津': ['北京', '河北'],
        '河北': ['北京', '天津', '山西', '内蒙古', '辽宁', '山东', '河南'],
        '山西': ['河北', '内蒙古', '陕西', '河南'],
        '内蒙古': ['黑龙江', '吉林', '辽宁', '河北', '山西', '陕西', '宁夏', '甘肃'],
        '辽宁': ['吉林', '河北', '内蒙古'], '吉林': ['黑龙江', '辽宁', '内蒙古'],
        '黑龙江': ['吉林', '内蒙古'], '上海': ['江苏', '浙江'],
        '江苏': ['山东', '安徽', '上海', '浙江'], '浙江': ['上海', '江苏', '安徽', '江西', '福建'],
        '安徽': ['江苏', '山东', '河南', '湖北', '江西', '浙江'], '福建': ['浙江', '江西', '广东'],
        '江西': ['安徽', '湖北', '湖南', '广东', '福建', '浙江'], '山东': ['河北', '河南', '安徽', '江苏'],
        '河南': ['河北', '山西', '陕西', '湖北', '安徽', '山东'], '湖北': ['河南', '陕西', '重庆', '湖南', '江西', '安徽'],
        '湖南': ['湖北', '重庆', '贵州', '广西', '广东', '江西'], '广东': ['福建', '江西', '湖南', '广西', '海南'],
        '广西': ['广东', '湖南', '贵州', '云南'], '海南': ['广东'], '重庆': ['陕西', '四川', '贵州', '湖南', '湖北'],
        '四川': ['重庆', '贵州', '云南', '西藏', '青海', '甘肃', '陕西'],
        '贵州': ['四川', '云南', '广西', '湖南', '重庆'], '云南': ['西藏', '四川', '贵州', '广西'],
        '西藏': ['新疆', '青海', '四川', '云南'], '陕西': ['内蒙古', '山西', '河南', '湖北', '重庆', '四川', '甘肃', '宁夏'],
        '甘肃': ['宁夏', '内蒙古', '陕西', '四川', '青海', '新疆'], '青海': ['新疆', '西藏', '四川', '甘肃'],
        '宁夏': ['陕西', '甘肃', '内蒙古'], '新疆': ['甘肃', '青海', '西藏']
    }

    def allowed(a, b):
        pa, pb = province_map.get(a), province_map.get(b)
        if pa is None or pb is None: return False
        if not is_all_prov and pa != selected_prov and pb != selected_prov: return False
        if scene == 1: return True
        if scene == 2: return pa == pb
        if scene == 3: return pa == pb or pb in neighbor_map.get(pa, []) or pa in neighbor_map.get(pb, [])
        return False

    dists = []
    for i in range(len(ctx.names)):
        for j in range(i + 1, len(ctx.names)):
            if not allowed(ctx.names[i], ctx.names[j]): continue
            d = haversine(ctx.lats[i], ctx.lons[i], ctx.lats[j], ctx.lons[j])
            if min_d <= d <= max_d:
                dists.append(d)
    if not dists:
        messagebox.showinfo("提示", "当前条件下无航线")
        return
    dists = np.array(dists)

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    plt.rcParams.update({'font.size': 15})

    sorted_d = np.sort(dists)
    cum_pct = np.arange(1, len(sorted_d)+1) / len(sorted_d) * 100
    ax1.plot(sorted_d, cum_pct, lw=3, color='steelblue')
    ax1.set_title("累计百分比曲线", fontsize=17, pad=15)
    ax1.set_xlabel("距离 km", fontsize=16)
    ax1.set_ylabel("≤该距离的航线占比 %", fontsize=16)
    ax1.grid(alpha=0.3)

    bin_w = 100
    left = (min_d // bin_w) * bin_w
    right = ((max_d // bin_w) + 1) * bin_w
    bins = np.arange(left, right + bin_w, bin_w)
    counts, edges = np.histogram(dists, bins=bins)
    ax2.bar(edges[:-1], counts, width=bin_w*0.9, align='edge', color='seagreen', edgecolor='black')
    ax2.set_title("每100 km 航线密度", fontsize=17, pad=15)
    ax2.set_xlabel("距离区间 km", fontsize=16)
    ax2.set_ylabel("航线数量", fontsize=16)
    for x, c in zip(edges[:-1] + bin_w/2, counts):
        if c > 0:
            ax2.text(x, c, int(c), ha='center', va='bottom', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

def count_transfer(ctx):
    try:
        rng = float(ctx.ent_range.get())
        if rng <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("输入错误", "请输入正数航程")
        return

    n = len(ctx.names)
    dist_mx = [[float('inf')]*n for _ in range(n)]
    for i in range(n):
        dist_mx[i][i] = 0
        for j in range(i+1, n):
            d = haversine(ctx.lats[i], ctx.lons[i], ctx.lats[j], ctx.lons[j])
            dist_mx[i][j] = dist_mx[j][i] = d

    cnt = floyd(dist_mx, rng)
    if sum(cnt) == 0:
        messagebox.showinfo("提示", "当前航程下无中转航线")
        return

    top = sorted(zip(ctx.names, cnt), key=lambda x: -x[1])[:10]
    labels, values = zip(*top)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color='coral')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("被当作中转机场的次数", fontsize=13)
    ax.set_title(f"飞机航程 {rng:.0f} km 时最繁忙的 10 个中转机场", fontsize=15)
    for i, v in enumerate(values):
        ax.text(v + 0.5, i, str(v), va='center', fontsize=12)
    plt.tight_layout()
    plt.show()

    map_win = tk.Toplevel(ctx.root)
    map_win.title("地图展示")
    map_win.geometry("280x80")
    tk.Label(map_win, text="是否把前十机场显示到中国地图？").pack(pady=10)
    def _yes():
        plot_top10_transfer_on_map(labels, ctx.df)
        map_win.destroy()
    tk.Button(map_win, text="显示地图", command=_yes).pack(side='left', padx=30)
    tk.Button(map_win, text="取消", command=map_win.destroy).pack(side='right', padx=30)

def floyd(dist_mx, rng):
    n = len(dist_mx)
    nxt = [[-1 if dist_mx[i][j] == float('inf') else j for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(n):
            if dist_mx[i][j] > rng:
                dist_mx[i][j] = float('inf')

    for k in range(n):
        for i in range(n):
            if dist_mx[i][k] == float('inf'):
                continue
            for j in range(n):
                if dist_mx[k][j] == float('inf'):
                    continue
                new = dist_mx[i][k] + dist_mx[k][j]
                if new < dist_mx[i][j]:
                    dist_mx[i][j] = new
                    nxt[i][j] = nxt[i][k]

    cnt = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j or dist_mx[i][j] == float('inf'):
                continue
            cur = i
            while cur != j:
                nxt_cur = nxt[cur][j]
                if nxt_cur == -1:
                    break
                if nxt_cur != j:
                    cnt[nxt_cur] += 1
                cur = nxt_cur
    return cnt

def plot_top10_transfer_on_map(top10_names, df):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_xlim(73, 136)
    ax.set_ylim(17, 55)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("前十繁忙中转机场分布")

    import cartopy.crs as ccrs
    import cartopy.feature as cfeat
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeat.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeat.BORDERS, linewidth=0.4)
    ax.set_extent([73, 136, 17, 55], crs=ccrs.PlateCarree())

    for rank, name in enumerate(top10_names):
        row = df[df['机场名字'] == name]
        if row.empty:
            continue
        lon, lat = row.iloc[0]['lon'], row.iloc[0]['lat']
        ax.plot(lon, lat, 'ro', markersize=9 - rank * 0.5)
        ax.text(lon + 0.5, lat + 0.5, str(rank + 1),
                transform=ccrs.PlateCarree(), fontsize=11, fontweight='bold')
    plt.show()
# ------------------ 6. 启动 ------------------
if __name__ == "__main__":
    excel_file = "机场信息调查.xlsx"
    if not os.path.exists(excel_file):
        messagebox.showerror("文件不存在", f"请将 Excel 文件命名为 {excel_file} 并放在脚本同目录")
    else:
        df = read_airports(excel_file)
        root = tk.Tk()
        app = AirportApp(root, df)
        root.mainloop()