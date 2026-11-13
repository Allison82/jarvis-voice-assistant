#!/bin/zsh
# 雙擊重新叫醒賈維斯（例如說過「賈維斯關機」之後）。平常開機登入就會自動啟動，不用點這個。
# 把 YOUR_USERNAME 換成你 launchd Label 用的名字（見 launchd/ 底下的 plist 範本）。
launchctl kickstart -k gui/$(id -u)/com.YOUR_USERNAME.jarvis && echo "賈維斯啟動中，約 10 秒後就可以跟它說話。可以關掉這個視窗。"
