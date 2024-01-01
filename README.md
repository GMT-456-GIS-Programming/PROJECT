# PROJECT
#Hello,
In this project, we first obtained a point cloud by using drone images and ground control points taken in the mine area. Afterwards, we obtained "dem" from the point cloud and performed various analyzes using "dem" data.

As you can see, we have drone images taken with WGS84 and 10 ground control point coordinates. It all started with this data.

We wrote a code by using the data we had and converting the data obtained with WGS84 into Turef Tm30 and obtained a point cloud. In this code, we used cv2, numpy, matplotlib, os and exifread libraries.
cv2: It is a library used in all image processing operations and the pioneer of this field.
numpy: NumPy (Numerical Python) is a mathematics library that allows us to perform scientific calculations quickly.
matplotlib: It is the basic Python library we use in data visualization. It allows us to make 2 and 3 dimensional drawings.
os: It is a module that allows us to easily perform operations on files and directories.
exifread: Used to read EXIF (Exchangeable image file format) data. EXIF is a standard that contains metadata information often found in digital photographs and images.

We obtained a dem (digital elevation model) with code using point cloud data. We used laspy, gdal and numpy in this code.
laspy: It is a library used to process and analyze LiDAR (Light Detection and Ranging) data.
gdal: GDAL (Geospatial Data Abstraction Library) is the library used in geographic information systems (GIS) applications and geographic data processing.
We wrote a code that reads the beam image we obtained, assigns colors according to the height, and visualizes it. In this code We used rasterio, numpy and matplotlib in this code.
rasterio: It is a library used to read, write and manipulate raster data in geographic information systems (GIS) applications.

Here you see demi and the analyzes obtained from dem. Slope, Colorful Slope, Apect and Height Groups.

Here and in the images following here you will see a comparison of the data we obtained and what was obtained with the application.

Here you see the slope analysis obtained with the code from dem. We analyzed this data on a range from 0 to 90. We used rasterio, numpy and matplotlib in this code.

Here you see the colored slope analysis obtained with the code from dem. We analyzed this data using 3 color scales. We used rasterio, numpy and matplotlib in this code.

Here you see the view analysis obtained with the code from dem. Aspect is a form of numerical land analysis used to determine the viewing directions of surfaces belonging to details found in the field. We used rasterio, numpy and matplotlib in this code.

Here you see the analysis of the height groups obtained with the code from dem. We analyzed the elevation of the area where the mining area is located relative to sea level. We used rasterio, numpy and matplotlib in this code.

We presented you the data and analysis we obtained with the codes. Our project had a process consisting of 6 parts. As you know, some mining lands harm green areas and nature. After the mine is out of use, it should be restored or greening plans should be made. These plans can be made much more efficiently by presenting these analyzes to agricultural engineers and landscape architects. Thank you for listening.
