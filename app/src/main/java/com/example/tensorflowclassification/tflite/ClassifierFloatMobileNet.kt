package com.example.tensorflowclassification.tflite

import android.app.Activity

class ClassifierFloatMobileNet(activity: Activity, device: Device, numThreads: Int ) : Classifier( activity, device, numThreads ) {
    companion object{
        const val IMAGE_MEAN: Float = 127.5f
        const val IMAGE_STD: Float = 127.5f
    }
    private var labelProbArray =  arrayOf( FloatArray(numLabels) )

    override fun getImageSizeX(): Int = 448
    override fun getImageSizeY(): Int = 448

    override fun getModelPath(): String = "mobilenet_v1_1.0_224.tflite"
    override fun getLabelPath(): String = "labels_mobilenet_v1_1.0_224.txt"
    override fun getNumberOfBytesPerChannel(): Int = 4

    override fun addPixelValue(pixelValue: Int) {
        imgData!!.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
        imgData!!.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
        imgData!!.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
    }

    override fun getProbability(labelIndex: Int): Float = labelProbArray[0][labelIndex]
    override fun setProbability(labelIndex: Int, value: Number) {
        labelProbArray[0][labelIndex] = value.toFloat()
    }

    override fun getNormalizedProbability(labelIndex: Int): Float = labelProbArray[0][labelIndex]

    override fun runInference() {
        tflite.run(imgData, labelProbArray)
    }
}