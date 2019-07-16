package com.example.tensorflowclassification.tflite

import android.app.Activity

class ClassifierQuantizedMobileNet( activity: Activity, device: Classifier.Device, numThreads: Int ): Classifier( activity, device, numThreads ) {
    private var labelProbArray: Array<ByteArray> =  arrayOf(ByteArray(numLabels))

    override fun getImageSizeX(): Int = 224
    override fun getImageSizeY(): Int = 224

    override fun getModelPath(): String = "mobilenet_v1_1.0_224_quant.tflite"
    override fun getLabelPath(): String = "labels_mobilenet_v1_1.0_224.txt"
    override fun getNumberOfBytesPerChannel(): Int = 4

    override fun addPixelValue(pixelValue: Int) {
        imgData!!.put((pixelValue shr 16 and 0xFF).toByte())
        imgData!!.put((pixelValue shr 8 and 0xFF).toByte())
        imgData!!.put((pixelValue and 0xFF).toByte())
    }

    override fun getProbability(labelIndex: Int): Float = labelProbArray[0][labelIndex].toFloat()
    override fun setProbability(labelIndex: Int, value: Number) {
        labelProbArray[0][labelIndex] = value.toByte()
    }

    override fun getNormalizedProbability(labelIndex: Int): Float = (labelProbArray[0][labelIndex].toInt() and 0xff) / 255.0f

    override fun runInference() {
        tflite.run(imgData, labelProbArray)
    }
}