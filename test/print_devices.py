import pyrealsense2 as rs
context = rs.context()
print("Connected devices:", [device.get_info(rs.camera_info.serial_number) for device in context.devices])
