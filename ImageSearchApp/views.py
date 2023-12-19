from django.shortcuts import render, loader
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import cv2
import glob
import os
import pickle
import csv
import pandas
from django.conf import settings
from pathlib import Path
from datetime import datetime
from io import BytesIO
from django.conf import settings
from subscriptable_path import Path as s_path
from .essnt_methods import EssentialMethodsClass
from .trademark_info import CompanyInfo

relative_path_profile = 'https://mdbcdn.b-cdn.net/img/new/avatars/1.webp'

class Home(TemplateView):
	template_name = 'AuthenticationApp:home.html'
	template_name = 'home.html'
	pageTitle = "Homepage"
	pageStatus = '1'
	homeActive = 'active'
	extra_context={'pageTitle': pageTitle, 'pageStatus': pageStatus, 'homekActive': homeActive,
                   'relative_path_profile': relative_path_profile,}

	
def index(request):
	indexActive = 'active'
	pageTitle = 'Greyscale'
	pageStatus = 1
	upload_dir = Path(str(settings.MEDIA_ROOT)+'/uploads/')
	if request.method == 'POST':
		uploaded_file = request.FILES['imagefile']
		pageStatus = 2
		# Save query image
		buffer = BytesIO()
		buffer.write(uploaded_file.read())
		buffer.seek(0)
		img = Image.open(buffer)  # PIL image
		uploaded_img_path_url = Path(str(upload_dir) + '/' +datetime.now().isoformat().replace(":", ".") + "_" + uploaded_file.name)
		img.save(uploaded_img_path_url)
		
		path = uploaded_img_path_url #FULL PATH
		start = settings.BASE_DIR
		relative_path = os.path.relpath(path, start)

		gambarGreyscale = img
		tujuan = s_path(uploaded_img_path_url)
		namafilebaru = tujuan[:8] + "_greyscale/" + tujuan[8:]
		filebaru = gambarGreyscale.convert(mode='L').save(namafilebaru)
		displayFileMod = relative_path[:13] + "_greyscale" + relative_path[13:]
		

		return render(request, 'grayscale.html', {
			'pageStatus':pageStatus,
			'displayFileMod':displayFileMod,
			'pageTitle':pageTitle,
			'indexActive':indexActive,
			'displayFile':namafilebaru,
			'url': uploaded_img_path_url,
			'settingsBASE_DIR': settings.BASE_DIR,
			'tujuan': tujuan,
			'settingsMEDI_DIR': settings.MEDIA_ROOT,
			'displayFile': relative_path,
			'gambarGreyscale': gambarGreyscale,
			'filebaru': filebaru,
			'relative_path_profile': relative_path_profile,
			})
	return render(request, 'grayscale.html', {
		'pageStatus':pageStatus,
		'pageTitle':pageTitle,
		'indexActive':indexActive,
		'settingsBASE_DIR': settings.BASE_DIR,
		'upload_dir': upload_dir,
		'settingsMEDI_DIR': settings.MEDIA_ROOT,
		'relative_path_profile':relative_path_profile,
		
		})



from .essnt_methods import EssentialMethodsClass
from .feature_ext_upload import ExtractFeatureUpload
esst_methods = EssentialMethodsClass()
query_image_obj = ExtractFeatureUpload()
root_dir = Path(str(settings.MEDIA_ROOT),'/Flickr_32')
upload_dir = Path(str(settings.MEDIA_ROOT)+'/uploads/')
features_dir = str(settings.MEDIA_ROOT)+'/feature/*'

filenames_new = sorted(query_image_obj.get_file_list())
length_of_new_file = len(filenames_new)
classList = query_image_obj.fileNamesOfData()
distance_info = query_image_obj.getDistanceInfo()
def performanceResult(request):
	seekActive = 'active'
	pageTitle = 'Image Seeker'
	pageStatus = 1
	positif = ['rose', 'sunf', 'tuli', 'dand', 'aste']
	actual = []
	pred = []
	datasetAccuracies = query_image_obj.calculate_accuracy() 
	datasetAccuracy = datasetAccuracies[0]
	precision = datasetAccuracies[1]
	recall = datasetAccuracies[2]
	f1score = datasetAccuracies[3]
	if request.method == 'POST':
		uploaded_file = request.FILES['imagefile']
		pageStatus = 2
		img = esst_methods.get_Uploaded_Image(uploaded_file)
		upload_dir = Path(str(settings.MEDIA_ROOT)+'/uploads/')
		uploaded_img_path = Path(str(upload_dir) + '/' +datetime.now().isoformat().replace(":", ".") + "_" + uploaded_file.name)
		img.save(uploaded_img_path)
		pathImage = uploaded_img_path
		displayFile = esst_methods.getrelativePathMediaTemplate(full_path=uploaded_img_path, exclude_path=esst_methods.base_dir)
		dataCounter = len(glob.glob1(Path(str(esst_methods.media_dir)+'/Flickr_32/**/'), "*.jpg"))
		
		queried_classNames = query_image_obj.getClassname(str(filenames_new[1]))		
		query_image = query_image_obj.extract_features(pathImage)
		k_neighbours = query_image_obj.knnMethod(query_image)
        
		splitClassName = []
		for filepath in k_neighbours:
			splitClassText = filepath.split('/')[-2]
			splitClassName.append(splitClassText)
    	
		zipped_list = zip(k_neighbours, splitClassName)
		

		return render(request,'performance.html', {
			'displayFile':displayFile,
			'pageStatus':pageStatus,
			'pageTitle':pageTitle,
			'seekActive':seekActive,
			'zipped_list': zipped_list,
			'relative_path_profile': relative_path_profile,
  			'queried_classNames': queried_classNames,
			# 'scores':scores,
			# 'nearest':nearest,
			# 'dataCounter':dataCounter,
			# 'accuracy':accuracy,
			'precision':precision,
			'recall':recall,
			'f1score':f1score
			})
	return render(request, 'performance.html', {
		'pageStatus':pageStatus,
		'pageTitle':pageTitle,
		'seekActive':seekActive,
  		'datasetAccuracy': datasetAccuracy,
    	'precision':precision,
		'recall':recall,
		'f1score':f1score,
		'root_dir': root_dir,
		'length_of_new_file': length_of_new_file,
		'classList': classList,
		'distance_info': distance_info,
		'filenames_new': filenames_new,
		'relative_path_profile': relative_path_profile,
		})



upload_dir = Path(str(settings.MEDIA_ROOT)+'/uploads/')
def rotate(request):
	rotateActive = 'active'
	pageTitle = 'Rotate'
	pageStatus = 1
	if request.method == 'POST':
		uploaded_file = request.FILES['imagefile']
		pageStatus = 2
		img = esst_methods.get_Uploaded_Image(uploaded_file)
		uploaded_img_path_url = Path(str(upload_dir) + '/' +datetime.now().isoformat().replace(":", ".") + "_" + uploaded_file.name)
		img.save(uploaded_img_path_url)
		displayFile = esst_methods.getrelativePathMediaTemplate(uploaded_img_path_url, settings.BASE_DIR)
		return render(request, 'rotate.html', {
			'displayFile':displayFile,
			'pageStatus':pageStatus,
			'pageTitle':pageTitle,
			'rotateActive':rotateActive,
			'image': img,
			'relative_path_profile':relative_path_profile
			})
	if request.GET.get('degree'):
		displayFile = request.GET['displayFromPallet']
		tujuan = settings.BASE_DIR + '/' + displayFile
		gambarRotate = Image.open(tujuan)
		namafilebaru = tujuan[:-4] + "_rotate" + tujuan[-4:]
		# CONVERT MULAI DISINI MENGGUNAKAN FUNGSI rotate()
		filebaru = gambarRotate.rotate(int(request.GET['degree']), expand=1).save(namafilebaru)
		displayFileMod = displayFile[:-4] + "_rotate" + displayFile[-4:]
		pageStatus = 3
		print(displayFile)
		print(displayFileMod)
		return render(request, 'rotate.html', {
			'displayFile':displayFile,
			'displayFileMod':displayFileMod,
			'pageStatus':pageStatus,
			'pageTitle':pageTitle,
			'rotateActive':rotateActive,
			'relative_path_profile': relative_path_profile,
			})
	return render(request, 'rotate.html', {
		'pageStatus':pageStatus,
		'pageTitle':pageTitle,
		'rotateActive':rotateActive,
		'relative_path_profile': relative_path_profile,
		})

def flip(request):
	flipActive = 'active'
	pageTitle = 'Flip'
	pageStatus = 1
	if request.method == 'POST':
		uploaded_file = request.FILES['imagefile']
		img = esst_methods.get_Uploaded_Image(uploaded_file)
		uploaded_img_path_url = Path(str(upload_dir) + '/' +datetime.now().isoformat().replace(":", ".") + "_" + uploaded_file.name)
		img.save(uploaded_img_path_url)
		displayFile = esst_methods.getrelativePathMediaTemplate(uploaded_img_path_url, settings.BASE_DIR)
		pageStatus = 2
		return render(request, 'flip.html', {
			'displayFile':displayFile,
			'pageStatus':pageStatus,
			'pageTitle':pageTitle,
			'flipActive':flipActive,
			'relative_path_profile':relative_path_profile
			})
	if request.GET.get('leftright'):
		displayFile = request.GET['displayFromPallet']
		# tujuan = "/srv/http/djangoproject" + displayFile
		tujuan = settings.BASE_DIR + '/' + displayFile
		gambarFlip = Image.open(tujuan)
		namafilebaru = tujuan[:-4] + "_flip" + tujuan[-4:]
		# CONVERT MULAI DISINI MENGGUNAKAN FUNGSI transpose()
		filebaru = gambarFlip.transpose(Image.FLIP_LEFT_RIGHT).save(namafilebaru)
		displayFileMod = displayFile[:-4] + "_flip" + displayFile[-4:]
		print(displayFile)
		print(displayFileMod)
		pageStatus = 3
		return render(request, 'flip.html', {
			'displayFile':displayFile,
			'displayFileMod':displayFileMod,
			'pageStatus':pageStatus,
			'pageTitle':pageTitle,
			'flipActive':flipActive,
			'relative_path_profile':relative_path_profile
			})
	if request.GET.get('topbottom'):
		displayFile = request.GET['displayFromPallet']
		# tujuan = "/srv/http/djangoproject" + displayFile
		tujuan = settings.BASE_DIR + '/' + displayFile
		gambarFlip = Image.open(tujuan)
		namafilebaru = tujuan[:-4] + "_flip" + tujuan[-4:]
		# CONVERT MULAI DISINI MENGGUNAKAN FUNGSI transpose()
		filebaru = gambarFlip.transpose(Image.FLIP_TOP_BOTTOM).save(namafilebaru)
		displayFileMod = displayFile[:-4] + "_flip" + displayFile[-4:]
		pageStatus = 3
		print(displayFile)
		print(displayFileMod)
		return render(request, 'flip.html', {
			'displayFile':displayFile,
			'displayFileMod':displayFileMod,
			'pageStatus':pageStatus,
			'pageTitle':pageTitle,
			'flipActive':flipActive,
			'relative_path_profile':relative_path_profile,
			})
	return render(request, 'flip.html', {
		'pageStatus':pageStatus,
		'pageTitle':pageTitle,
		'flipActive':flipActive,
		'relative_path_profile':relative_path_profile
		})

def crop(request):
	cropActive = 'active'
	pageTitle = 'Crop'
	pageStatus = 1
	if request.method == 'POST':
		uploaded_file = request.FILES['imagefile']
		pageStatus = 2
		img = esst_methods.get_Uploaded_Image(uploaded_file)
		uploaded_img_path_url = Path(str(upload_dir) + '/' +datetime.now().isoformat().replace(":", ".") + "_" + uploaded_file.name)
		img.save(uploaded_img_path_url)
		displayFile = esst_methods.getrelativePathMediaTemplate(uploaded_img_path_url, settings.BASE_DIR)
		return render(request, 'crop.html', {
			'displayFile':displayFile,
			'pageStatus':pageStatus,
			'pageTitle':pageTitle,
			'cropActive':cropActive,
			'relative_path_profile':relative_path_profile
			})
	if request.GET.get('x'):
		displayFile = request.GET['displayFromPallet']
		# tujuan = "/srv/http/djangoproject" + displayFile
		tujuan = settings.BASE_DIR + '/' + displayFile
		gambarCrop = Image.open(tujuan)
		namafilebaru = tujuan[:-4] + "_crop" + tujuan[-4:]
		# CONVERT MULAI DISINI MENGGUNAKAN FUNGSI crop()
		filebaru = gambarCrop.crop((float(request.GET['x']), float(request.GET['y']), float(request.GET['w'])+float(request.GET['x']), float(request.GET['h'])+float(request.GET['y'])) ).save(namafilebaru)
		displayFileMod = displayFile[:-4] + "_crop" + displayFile[-4:]
		pageStatus = 3
		print(displayFile)
		print(displayFileMod)
		return render(request, 'crop.html', {
			'displayFile':displayFile,
			'displayFileMod':displayFileMod,
			'pageStatus':pageStatus,
			'pageTitle':pageTitle,
			'cropActive':cropActive,
			'relative_path_profile':relative_path_profile
			})
	return render(request, 'crop.html', {
		'pageStatus':pageStatus,
		'pageTitle':pageTitle,
		'cropActive':cropActive,
		'relative_path_profile':relative_path_profile
		})

def scale(request):
	scaleActive = 'active'
	pageTitle = 'Scale'
	return render(request, 'scale.html', {
		'pageTitle':pageTitle,
		'scaleActive':scaleActive
		})

def invert(request):
	invertActive = 'active'
	pageTitle = 'Invert'
	return render(request, 'invert.html', {
		'pageTitle':pageTitle,
		'invertActive':invertActive
		})

from . essnt_methods import WebScraping
from .feature_ext_upload import ExtractFeatureUpload
query_image_obj = ExtractFeatureUpload()
query_image_obj1 = ExtractFeatureUpload()
relfilenames = query_image_obj.fileNamesOfData()
filenames_length = len(relfilenames)
featureAttr = query_image_obj.featureListAttributes()
classNames = query_image_obj.fileNamesOfData()
company_info_obj = CompanyInfo()

def searchFlickrData(request):
    # query_image_obj1 = ExtractFeatureUpload()
    searchActive = 'active'
    pageTitle = 'Image Search'
    pageStatus = 1
    mixedimages = []
    k_neighbours = []
    quitesimilar = []
    exactsimilar = []
    exactclassnames = []
    quitesimilarclassnames = []
    
    upload_dir = Path(str(settings.MEDIA_ROOT)+'/uploads/')
    root_dir = Path(str(settings.MEDIA_ROOT)+'/Flickr_32')
    if request.method == 'POST' and request.FILES['imagefile']:
        image = request.FILES['imagefile']
        pageStatus = 2
        # Save query image
        from io import BytesIO
        buffer = BytesIO()
        buffer.write(image.read())
        buffer.seek(0)
        img = Image.open(buffer)  # PIL image
        uploaded_img_path = Path(str(upload_dir) + '/' +datetime.now().isoformat().replace(":", ".") + "_" + image.name)
        img.save(uploaded_img_path)

        path = uploaded_img_path
        start = settings.BASE_DIR
        uploaded_img_rel_path = os.path.relpath(path, start)

        query_image = query_image_obj1.extract_features(path)
        k_neighbours = query_image_obj1.knnMethod(query_image)
        mixedimages = query_image_obj1.findExactSimilar(k_neighbours)
        exactsimilar = mixedimages['exactsimilar']
        quitesimilar = mixedimages['quitsimilar']
        exactclassnames = mixedimages['exactclassnames']
        quitesimilarclassnames = mixedimages['quitesimilarclassnames']
        notmatchfound = len(exactsimilar)
        if notmatchfound < 3:
            notmatchfound = False
        else:
            notmatchfound = True
             
        companyInfo_list = company_info_obj.Infolist(exactclassnames)
        brand_info_link = company_info_obj.brand_Info(exactclassnames)
        # queried_accuracy = company_info_obj.queriedAccuracy(splitClassName)
        
        exactURL = WebScraping.googleScraping(exactclassnames[0])
        brand_info = WebScraping.wikipediaScraping(exactclassnames[0])
        exactURList = WebScraping.getBrandURL(exactclassnames[0])
        
        zipped_list = zip(exactsimilar, exactclassnames)
        zipped_list_quitesimilar = zip(quitesimilar, quitesimilarclassnames)

        return render(request, 'imagesearch.html', 
                  {'uploaded_img_path' : uploaded_img_rel_path, 'query_image_feature' : query_image,
                    'filenames': relfilenames, 'filenames_length': filenames_length, 'k_neighbours':k_neighbours,
                    'searchActive': searchActive, 'pageTitle': pageTitle, 'pageStatus': pageStatus, 'zipped_list':zipped_list, 
                    'brand_info_link': brand_info_link, 'relative_path_profile': relative_path_profile,
                    'companyInfo_list': companyInfo_list, 
                    'exactURL':exactURL, 
                    'quitesimilar':quitesimilar,
                    'brand_info':brand_info, 
                    'exactURList':exactURList,
                    'exactsimilar':exactsimilar, 'quitesimilarclassnames':quitesimilarclassnames,
                    'exactclassnames':exactclassnames,
                    'zipped_list_quitesimilar':zipped_list_quitesimilar,
                    'notmatchfound':notmatchfound,})

    else:
        return render(request, 'imagesearch.html',{'media_root': settings.MEDIA_ROOT,
                                              'root_dir': root_dir, 'filenames': relfilenames,
                                             'filenames_length': filenames_length, 'featureAttr0': featureAttr[0],
                                               'featureAttr1': featureAttr[1], 'featureAttr2': featureAttr[2],
                                               'classNames': classNames, 'relative_path_profile': relative_path_profile,
                                               'searchActive':searchActive, 'pageTitle': pageTitle, 'pageStatus': pageStatus} )
from .essnt_methods import ConvertCSV
def textualSearch(request):
    searchActive = 'active'
    pageTitle = 'Image Search'
    pageStatus = 1
    if request.method == 'POST':
        pageStatus = 2
        searched_data = str(request.POST.get('searchbox'))
        image_url = ConvertCSV.searchURLImage(name=searched_data)
        brand_names = ConvertCSV.searchBrandsName(name= searched_data)
        if image_url and brand_names:
            return render(request, 'textsearch.html', {'brand_names':brand_names, 'image_url':image_url, 'searchActive':searchActive,
                                                               'pageTitle':pageTitle, 'pageStatus':pageStatus,
                                                               'relative_path_profile':relative_path_profile})
        else:
            return render(request, 'textsearch.html', { })
    else:
        return render(request, 'textsearch.html', { 'searchActive':searchActive,
                                                               'pageTitle':pageTitle, 'pageStatus':pageStatus,
                                                               'relative_path_profile':relative_path_profile})
    
   
def textbasedsearch(request):
    textsearchActive = 'active'
    pageTitle = 'Image Search'
    pageStatus = 1
    ifo_box = 'Search the Brand Images'
    if request.method == 'POST':
        pageStatus = 2
        searched_data = str(request.POST.get('fullsearchbox'))
        image_url = ConvertCSV.searchURLImage(name=searched_data)
        brand_names = ConvertCSV.searchBrandsName(name= searched_data)
        if image_url and brand_names:
            return render(request, 'textbasedsearch.html', {'brand_names':brand_names, 'image_url':image_url, 'textsearchActive':textsearchActive,
                                                               'pageTitle':pageTitle, 'pageStatus':pageStatus,
                                                               'relative_path_profile':relative_path_profile})
        else:
            return render(request, 'textbasedsearch.html', { })
    else:
        return render(request, 'textbasedsearch.html', { 'ifo_box':ifo_box, 'textsearchActive':textsearchActive,
                                                               'pageTitle':pageTitle, 'pageStatus':pageStatus,
                                                               'relative_path_profile':relative_path_profile})

from .essnt_methods import EssentialMethodsClass, RandomValues
def topCompanyList(request):
    topcompanyActive = 'active'
    pageTitle = 'Top Search'
    pageStatus = 1
    randObj = RandomValues()
    data_list = randObj.readFile()
    random_list = randObj.randomizeList(data_list)
    zipList = zip(random_list[0], random_list[1])
    return render(request, 'topcompany.html', {'random_list':zipList,'topcompanyActive':topcompanyActive,
                                                               'pageTitle':pageTitle, 'pageStatus':pageStatus,})
            
from .featureExtraction150 import ExtractFeatureUpload150        
def testPageFor150(request):
    obj_150 = ExtractFeatureUpload150()
    filenameslist = obj_150.filenames
    relativefilelist = obj_150.fileNamesOfData()
    category = obj_150.categoryList()
    wholedataclassnames = obj_150.wholeDataClassNames()
    return HttpResponse(f'FILE NAME LIST ==> {filenameslist} /n RELATIVE FILE LIST ==> {relativefilelist}  BRAND NAME LIST ==> {category}/n WHOLE DATA CLASSNAME ==> {wholedataclassnames}')
    
def searchBigData150(request):
    searchActive = 'active'
    pageTitle = 'Image Search'
    pageStatus = 1
    
    upload_dir = Path(str(settings.MEDIA_ROOT)+'/uploads/')
    root_dir = Path(str(settings.MEDIA_ROOT)+'/Flickr_32')
    if request.method == 'POST' and request.FILES['imagefile']:
        image = request.FILES['imagefile']
        pageStatus = 2
        # Save query image
        from io import BytesIO
        buffer = BytesIO()
        buffer.write(image.read())
        buffer.seek(0)
        img = Image.open(buffer)  # PIL image
        uploaded_img_path = Path(str(upload_dir) + '/' +datetime.now().isoformat().replace(":", ".") + "_" + image.name)
        img.save(uploaded_img_path)

        path = uploaded_img_path
        start = settings.BASE_DIR
        uploaded_img_rel_path = os.path.relpath(path, start)

        query_image = query_image_obj.extract_features(path)
        k_neighbours = query_image_obj.knnMethod(query_image)
        
        splitClassName = []
        for filepath in k_neighbours:
            splitClassText = filepath.split('/')[-2]
            splitClassName.append(splitClassText)
             
        companyInfo_list = company_info_obj.Infolist(splitClassName)
        brand_info_link = company_info_obj.brand_Info(splitClassName)
        queried_accuracy = company_info_obj.queriedAccuracy(splitClassName)
        
        exactURL = WebScraping.googleScraping(splitClassName[1])
        brand_info = WebScraping.wikipediaScraping(splitClassName[1])
        exactURList = WebScraping.getBrandURL(splitClassName[1])
        
        zipped_list = zip(k_neighbours, splitClassName)
        
        return render(request, 'imagesearch.html', 
                  {'uploaded_img_path' : uploaded_img_rel_path, 'query_image_feature' : query_image,
                    'filenames': relfilenames, 'filenames_length': filenames_length, 'k_neighbours':k_neighbours,
                    'searchActive': searchActive, 'pageTitle': pageTitle, 'pageStatus': pageStatus, 'zipped_list':zipped_list, 
                    'brand_info_link': brand_info_link, 'relative_path_profile': relative_path_profile,
                    'companyInfo_list': companyInfo_list, 'queried_accuracy': queried_accuracy, 'exactURL':exactURL,
                    'brand_info':brand_info, 'exactURList':exactURList,})

    else:
        return render(request, 'imagesearch.html',{'media_root': settings.MEDIA_ROOT,
                                              'root_dir': root_dir, 'filenames': relfilenames,
                                             'filenames_length': filenames_length, 'featureAttr0': featureAttr[0],
                                               'featureAttr1': featureAttr[1], 'featureAttr2': featureAttr[2],
                                               'classNames': classNames, 'relative_path_profile': relative_path_profile,
                                               'searchActive':searchActive, 'pageTitle': pageTitle, 'pageStatus': pageStatus} )
      
     
    
	