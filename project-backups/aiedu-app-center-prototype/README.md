# aiedu.haoduo.vip/apps 本地原型

## 本地内容

- `source/`：从线上站点抓取的页面壳、关键前端资源、应用中心接口数据和视觉素材。
- `references/source-apps-page.png`：线上页面渲染截图，用作视觉参考。
- `prototype/`：基于真实接口数据制作的静态高保真原型。

## 已确认的信息

- 页面入口：`https://aiedu.haoduo.vip/apps`
- 前端资源域名：`https://assets.haoduo.vip`
- 应用中心数据接口：`https://ai-apps-api.haoduo.vip/api/app-center/sections`
- 页面结构：顶部导航 + 应用中心分区列表，当前包含 `AI教学工具`、`AI实验室`、`学科智能体/智能应用` 三个分区。

## 打开方式

在当前目录启动一个静态服务后访问：

```bash
python3 -m http.server 4173
```

然后打开：

```text
http://127.0.0.1:4173/prototype/
```
