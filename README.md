# python_info

```sh
cd python-info
```
```sh
python -m venv .venv
```
```sh
source .venv/bin/activate
```
```sh
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

```sh
python src/main.py
```


<!-- Установка ClickHouse

# Устанавливаем необходимые пакеты
sudo dnf install -y epel-release
sudo dnf install -y yum-utils

# Добавляем репозиторий ClickHouse
sudo yum-config-manager --add-repo https://packages.clickhouse.com/rpm/clickhouse.repo

# Устанавливаем
sudo dnf install -y clickhouse-server clickhouse-client

# Запускаем сервис
sudo systemctl enable clickhouse-server
sudo systemctl start clickhouse-server

# Проверяем статус
sudo systemctl status clickhouse-server

# Тестируем подключение
clickhouse-client --query "SELECT version()"


sudo firewall-cmd --zone=public --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" port port="9000" protocol="tcp" accept' --permanent
sudo firewall-cmd --zone=public --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" port port="8123" protocol="tcp" accept' --permanent


sudo firewall-cmd --reload

bash
sudo nano /etc/clickhouse-server/config.xml

Найдите и раскомментируйте или добавьте:
<listen_host>0.0.0.0</listen_host>
