#!/bin/bash
# WARNING: this script is potentially dangerous, it modifies the authorizationdb
# in a way that is far from best practice. your funeral

APP_PROCESS_NAME="TeamViewer"

DAEMON_PATHS=( "/Library/LaunchDaemons/com.teamviewer.Helper.plist"
               "/Library/LaunchDaemons/com.teamviewer.teamviewer_service.plist" )

REMOVE_PATHS=( "/Applications/TeamViewer.app"
               "/Library/LaunchDaemons/com.teamviewer.Helper.plist"
               "/Library/LaunchDaemons/com.teamviewer.teamviewer_service.plist"
               "/Library/LaunchAgents/com.teamviewer.teamviewer.plist"
               "/Library/LaunchAgents/com.teamviewer.teamviewer_desktop.plist"
               "/Library/Security/SecurityAgentPlugins/TeamViewerAuthPlugin.bundle"
               "/Library/PrivilegedHelperTools/com.teamviewer.Helper"
               "/Library/Fonts/TeamViewer11.otf" )

for daemonPath in "${DAEMON_PATHS[@]}"
do
    if [ -e "$daemonPath" ]; then
        launchctl unload "$daemonPath"
    else
        printf "Not found: %s\n" "$daemonPath"
    fi
done

if pgrep "$APP_PROCESS_NAME"; then
    killall "$APP_PROCESS_NAME"
fi

for removePath in "${REMOVE_PATHS[@]}"
do
    if [ -e "$removePath" ]; then
        printf "Deleting: %s\n" "$removePath"
        rm -rf "$removePath"
    else
        printf "Not found: %s\n" "$removePath"
    fi
done

# this part is dangerous and not a good way to do this
# shout out to thevoices for providing corrections to this section
# using sed to make edits to authdb is frowned upon
authdbTeamViewer=$(security authorizationdb read system.login.console 2>/dev/null | grep TeamViewerAuthPlugin)
if [[ "$authdbTeamViewer" != "" ]]; then
    security authorizationdb read system.login.console > /tmp/authdb-system.login.console.plist
    sed -i.bak "/<string>TeamViewerAuthPlugin:start<\/string>/d" /tmp/authdb-system.login.console.plist
    security authorizationdb write system.login.console < /tmp/authdb-system.login.console.plist
fi
