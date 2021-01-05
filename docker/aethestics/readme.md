安裝docker image, container, DB
```
$ docker-compose --project-directory . -f docker/dev/docker-compose.yaml up -d
$ docker run --rm -v ${PWD}/src:/app -w /app composer install
$ docker exec -it test_pgsql_1 psql -U postgres
postgres=# create database test;
```
網域設定
```
$ sudo vi /etc/hosts
```
在 127.0.0.1 行末加入 test.localhost
