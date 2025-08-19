FROM alpine:latest

RUN apk add --no-cache curl bash docker

RUN URL=$(curl -s https://api.github.com/repos/nuclio/nuclio/releases/latest \
        | grep -i "browser_download_url.*nuctl.*linux-amd64" \
        | cut -d : -f 2,3 \
        | tr -d \" \
        | tr -d ' ') && \
        wget -O nuctl "$URL" && chmod +x nuctl

RUN mv nuctl /usr/local/bin/nuctl

ENTRYPOINT ["nuctl"]

CMD ["--help"]