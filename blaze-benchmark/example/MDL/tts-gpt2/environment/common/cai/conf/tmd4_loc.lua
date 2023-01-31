-- common functions

tmd4_loc = {}

function tmd4_loc.do_auto_cc ()
        ngx.status = 403
        local sec = ngx.var.tmd_seccookie
        local secname = ngx.var.tmd_sec_cookie_name

        if sec == "" or sec == nil or secname == "" or secname == nil then
            local content = "You have requested too frequently"
            ngx.print(content)
            ngx.exit(200)
            return
        end

        ngx.header["Set-Cookie"] = string.format([[%s=%s;path=/;HttpOnly]], secname, sec)
        ngx.header.cache_control = "no-cache"
        local content=[[<html>You have requested too frequently<script>document.location.reload();</script></html>]]
        ngx.print(content)
        ngx.exit(200)
end

function tmd4_loc.punish_url ()
        return string.format("http://%s/deny.html", ngx.var.tmd_domain)
end

function tmd4_loc.wait_url ()
        local scheme = ngx.var.scheme

        if ngx.var.http_X_Client_Scheme == "https" then
            scheme = "https"
        end

        return string.format("http://%s/wait.html?id=q&app=%s&wait_time=30&http_referer=%s://%s%s?",
        ngx.var.tmd_domain, ngx.var.tmd_app, scheme, ngx.var.host, ngx.var.request_uri)
end

function tmd4_loc.sm_cc_url (ip, sign_ip, owner)
        local scheme = ngx.var.scheme

        if ngx.var.http_X_Client_Scheme == "https" then
            scheme = "https"
        end

        return string.format("http://%s/checkcodev3.php?v=4&ip=%s&sign=%s&app=%s&how=%s&http_referer=%s://%s%s?",
        ngx.var.tmd_domain, ip, sign_ip, ngx.var.tmd_app,
        owner, scheme, ngx.var.host, ngx.var.request_uri)
end

function tmd4_loc.sm_login_url (v, u)
        return string.format("https://login.taobao.com/member/login.jhtml?from=smtmd-%s&redirectURL=%s?", v, u)
end

function tmd4_loc.do_action (redirect_loc)

    if ngx.var.http_X_Requested_With == "XMLHttpRequest" then
        tmd4_loc.xhr_request()
        return 1;
    elseif ngx.var.arg_callback ~= nil then
        tmd4_loc.callback_request()
        return 1;
    elseif redirect_loc ~= nil then
        ngx.header.cache_control = "no-cache"
        ngx.redirect(redirect_loc)
    end

    return 0;
end

function tmd4_loc.callback_request ()
    ngx.header.content_type = 'application/json'
    ngx.header.cache_control = 'no-cache'

    ngx.print(ngx.var['arg_callback'] .. '({"status":1111,"wait":5})')
end

function tmd4_loc.xhr_request ()
    local accept = ngx.var.http_accept
    ngx.header.cache_control = 'no-cache'

    if string.match(accept, 'application/xml') then
        ngx.header.content_type = 'application/xml'
        ngx.print("<root><status>1111</status><wait>5</wait></root>")
    elseif string.match(accept, 'text/plain') then
        ngx.header.content_type = 'text/plain'
        ngx.print("System busy now, pleasy try it again later.")
    elseif string.match(accept, 'text/html') then
        ngx.header.content_type = 'text/html'
        ngx.print("System busy now, pleasy try it again later.")
    else
        -- json request
        ngx.header.content_type = 'application/json'
        ngx.print('{"status":1111,"wait":5}')
    end
end

return tmd4_loc