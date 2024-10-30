!/bin/bash

# create the service file
echo "[Unit]
Description=Detects laughter, creating a counter in csv format
After=network.target

[Service]
Type=idle
Restart=on-failure
User=tec
ExecStart=/bin/bash -c 'cd /home/tec/Desktop/NewTestModels/joy/ && source venv1/bin/activate && python3 sound_event_detection.py >> detection_log.csv'

[Install]
WantedBy=multi-user.target" > /lib/systemd/system/laugh_detector.service

# update permissions
chmod 644 /lib/systemd/system/laugh_detector.service

# launch the new service
systemctl daemon-reload
systemctl enable laugh_detector.service

