"""登录闸与账号接口的端到端测试。

这些必须走真实的 HTTP 栈（TestClient）而不是直接调函数：要验的正是
中间件、Cookie 和状态码这几层，绕过它们等于什么都没测。

用 LCM90_DATA_DIR 把账号文件引到临时目录 —— 测试绝不能碰开发机上
真实的 data/users.json，否则跑一次测试就把自己的密码改掉了。
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.mkdtemp()
    os.environ["LCM90_DATA_DIR"] = tmp
    import app.main as main
    importlib.reload(main)          # 重载才能让新的环境变量生效
    with TestClient(main.app) as c:
        yield c
    main.session.shutdown()
    os.environ.pop("LCM90_DATA_DIR", None)


def _login(client, username="admin", password="admin"):
    return client.post("/api/login",
                       json={"username": username, "password": password})


# ---------------------------------------------------------------- 登录闸

def test_api_requires_login(client):
    client.cookies.clear()
    r = client.get("/api/status")
    assert r.status_code == 401, "未登录访问接口必须 401，实际 %d" % r.status_code
    assert "登录" in r.json().get("detail", "")


def test_video_stream_requires_login(client):
    """/video 是实时画面，漏在登录闸外面等于门没关。"""
    client.cookies.clear()
    r = client.get("/video", follow_redirects=False)
    assert r.status_code in (302, 401), \
        "未登录时 /video 必须被拦，实际 %d" % r.status_code


def test_page_redirects_to_login(client):
    client.cookies.clear()
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/login"


def test_login_page_is_public(client):
    client.cookies.clear()
    r = client.get("/login")
    assert r.status_code == 200 and "登录" in r.text


def test_wrong_password_rejected(client):
    client.cookies.clear()
    r = _login(client, "admin", "不是这个密码")
    assert r.status_code == 401
    # 提示不能区分"用户不存在"和"密码错"，否则可以拿来枚举账号
    assert r.json()["detail"] == _login(client, "查无此人", "x").json()["detail"]


def test_login_then_access(client):
    client.cookies.clear()
    r = _login(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["role"] == "admin" and body["is_default_password"] is True

    st = client.get("/api/status")
    assert st.status_code == 200
    data = st.json()
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "admin"
    assert data["user"]["is_default_password"] is True, \
        "还是出厂密码时必须如实上报，界面靠它弹警告"
    assert "record" in data, "status 里要带录像状态，界面靠它显示录制中"


def test_logout_invalidates_session(client):
    client.cookies.clear()
    _login(client)
    assert client.get("/api/status").status_code == 200
    assert client.post("/api/logout").status_code == 200
    assert client.get("/api/status").status_code == 401, "退出后必须失效"


# ---------------------------------------------------------------- 账号管理

def test_admin_can_manage_users_and_normal_user_cannot(client):
    client.cookies.clear()
    _login(client)

    r = client.post("/api/users", json={"username": "student",
                                        "password": "stupass",
                                        "role": "user"})
    assert r.status_code == 200, r.text

    names = [u["username"] for u in client.get("/api/users").json()["users"]]
    assert "student" in names and "admin" in names

    # 重名要拦
    assert client.post("/api/users", json={"username": "student",
                                           "password": "x2345"}).status_code == 400
    # 非法用户名要拦
    assert client.post("/api/users", json={"username": "有中文",
                                           "password": "x2345"}).status_code == 400
    # 太短的密码要拦
    assert client.post("/api/users", json={"username": "shorty",
                                           "password": "1"}).status_code == 400

    # 换成普通账号：能用软件，但碰不了账号管理
    client.cookies.clear()
    assert _login(client, "student", "stupass").status_code == 200
    assert client.get("/api/status").status_code == 200, "普通账号应能正常用软件"
    assert client.get("/api/users").status_code == 403
    assert client.post("/api/users", json={"username": "x1", "password": "p123"}
                       ).status_code == 403

    # admin 删掉它
    client.cookies.clear()
    _login(client)
    assert client.delete("/api/users/student").status_code == 200
    assert "student" not in [u["username"]
                            for u in client.get("/api/users").json()["users"]]


def test_cannot_delete_self_or_last_admin(client):
    client.cookies.clear()
    _login(client)
    r = client.delete("/api/users/admin")
    assert r.status_code == 400 and "自己" in r.json()["detail"]


def test_change_password_flow(client):
    client.cookies.clear()
    _login(client)

    # 原密码不对要拒绝
    r = client.post("/api/password", json={"old_password": "错的",
                                           "new_password": "brandnew"})
    assert r.status_code == 400 and "原密码" in r.json()["detail"]

    r = client.post("/api/password", json={"old_password": "admin",
                                           "new_password": "brandnew"})
    assert r.status_code == 200, r.text
    # 改完密码当前浏览器不该被踢下线（后端换发了新票）
    assert client.get("/api/status").status_code == 200, "改完密码不该把自己踢出去"

    st = client.get("/api/status").json()
    assert st["user"]["is_default_password"] is False, "改完不该再报出厂密码"

    # 旧密码作废、新密码可用
    client.cookies.clear()
    assert _login(client, "admin", "admin").status_code == 401
    assert _login(client, "admin", "brandnew").status_code == 200

    # 改回去，不影响同模块里后面的用例
    client.post("/api/password", json={"old_password": "brandnew",
                                       "new_password": "admin"})


# ---------------------------------------------------------------- 录像

def test_record_requires_login(client):
    client.cookies.clear()
    assert client.post("/api/record/start").status_code == 401


def test_record_without_camera_explains_why(client):
    client.cookies.clear()
    _login(client)
    r = client.post("/api/record/start")
    # 测试环境没有相机：必须给出人话原因，而不是 500
    assert r.status_code == 400, "没画面时应 400 并说明原因，实际 %d" % r.status_code
    assert "视频源" in r.json()["detail"]

    # 没在录的时候点停止不该报错
    r = client.post("/api/record/stop")
    assert r.status_code == 200 and r.json()["ok"] is True
