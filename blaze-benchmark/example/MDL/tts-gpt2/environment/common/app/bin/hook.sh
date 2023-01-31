# add hooks for appctl.sh when app start, stop.
# apps can add custom actions in these functions, default are empty.

# before application server (tomcat) start, jvm process not exists.
beforeStartApp() {
	  return
}

# after application server (tomcat) start, localhost:7001 is ready.
afterStartApp() {
	  return
}

# before application server (tomcat) stop, localhost:7001 is available.
beforeStopApp() {
	  return
}

# after application server (tomcat) stop, jvm process has exited.
afterStopApp() {
	  return
}