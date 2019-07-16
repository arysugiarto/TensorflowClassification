package com.example.tensorflowclassification.tflite

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.ArrayList


abstract class Classifier protected constructor(val activity: Activity, val device: Device, val numThreads: Int) {
    enum class Model {
        FLOAT,
        QUANTIZED
    }

    enum class Device {
        CPU,
        NNAPI,
        GPU
    }

    private val intValues: Array<Int> = Array<Int>(getImageSizeX() * getImageSizeY()) { 0 }

    private val tfLiteOptions: Interpreter.Options = Interpreter.Options()

    private lateinit var tfliteModel: MappedByteBuffer

    private var labels: List<String>

    private var gpuDelegate: GpuDelegate? = null

    protected var tflite: Interpreter

    protected var imgData: ByteBuffer? = null

    companion object {

        const val MAX_RESULTS = 3
        const val DIM_BATCH_SIZE = 1
        const val DIM_PIXEL_SIZE = 3

        fun create(activity: Activity, model: Model, device: Device, numThreads: Int): Classifier =
            if (model == Model.QUANTIZED) {
                ClassifierQuantizedMobileNet( activity, device, numThreads )
            } else {
                ClassifierFloatMobileNet( activity, device, numThreads )
            }

        class Recognition(val id: String?, val title: String?, val confidence: Float?, val location: RectF?) {
            override fun toString(): String {
                var resultString = ""
                resultString = if (id != null) "$resultString[$id!!] " else resultString
                resultString = if (title != null) "$resultString$title!! " else resultString
                resultString = if (confidence != null) "$resultString${String.format(
                    "(%.1f%%) ",
                    confidence * 100.0f
                )} " else resultString
                resultString = if (location != null) "$resultString$location " else resultString
                return resultString
            }
        }
    }

     init {
        val tfliteModel = loadModelFile(activity)
        when(device) {
            Device.NNAPI -> tfLiteOptions.setUseNNAPI(true)
            Device.GPU -> {
                gpuDelegate = GpuDelegate()
                tfLiteOptions.addDelegate(gpuDelegate)
            }
            Device.CPU -> { }
        }
        tfLiteOptions.setNumThreads(numThreads)
        tflite = Interpreter(tfliteModel, tfLiteOptions)
        labels = loadLabelList(activity)
        val imgData = ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE
                * getImageSizeX()
                * getImageSizeY()
                * DIM_PIXEL_SIZE
                * getNumberOfBytesPerChannel()
        )
        imgData.order(ByteOrder.nativeOrder())
    }

    private fun loadModelFile( activity: Activity ): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(getModelPath())
        val inputStream = FileInputStream( fileDescriptor.fileDescriptor )
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabelList(activity: Activity) : List<String> {
        val labels = mutableListOf<String>()
        val reader = BufferedReader( InputStreamReader( activity.assets.open(getLabelPath()) ) )
        var line: String? = reader.readLine();
        while(line != null) {
            labels.add(line);
            line = reader.readLine()
        }
        reader.close()
        return labels.toList()
    }

    private fun convertImageToByteBuffer( image: Bitmap ): ByteBuffer {
        val imgDataLocal = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * getImageSizeX() * getImageSizeY() * DIM_PIXEL_SIZE)
        val intValues = IntArray(getImageSizeX() * getImageSizeY())
        imgDataLocal.order(ByteOrder.nativeOrder())
        image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
        var pixel = 0;
        for (i in 0 until getImageSizeX()) {
            for (j in 0 until getImageSizeY()) {
                val value = intValues[pixel]
                imgDataLocal.put((value shr 16 and 0xFF).toByte())
                imgDataLocal.put((value shr 8 and 0xFF).toByte())
                imgDataLocal.put((value and 0xFF).toByte())
                pixel++
            }
        }
        return imgDataLocal
    }

    fun recognizeImage(image: Bitmap): List<Recognition>{
        imgData = convertImageToByteBuffer(Bitmap.createScaledBitmap(image, getImageSizeX(), getImageSizeY(), false))
        runInference()
        val pq: PriorityQueue<Recognition> = PriorityQueue(
            3,
            kotlin.Comparator { lhs, rhs -> compareValues<Comparable<Float>>( rhs.confidence!!, lhs.confidence!! ) }
        )

        (0 until labels.size).toList()
            .forEach {
                pq.add(
                    Recognition(
                    "$it",
                    if (labels.size > it) labels.get(it) else "unknown",
                    getNormalizedProbability(it),
                    null
                ) )
            }

        val recognitionSize = Math.min( pq.size, MAX_RESULTS )
        val recognitions = ArrayList<Recognition>()
        for ( i in 0 until recognitionSize ) recognitions.add( pq.poll() )
        return recognitions.toList()
    }

    fun close() {
        tflite.close()
        gpuDelegate?.close()
    }

    abstract fun getImageSizeX(): Int

    abstract fun getImageSizeY(): Int

    protected abstract fun getModelPath(): String

    protected abstract fun getLabelPath(): String

    protected abstract fun getNumberOfBytesPerChannel(): Int

    protected abstract fun addPixelValue(pixelValue: Int)

    protected abstract fun getProbability(labelIndex: Int): Float
    protected abstract fun setProbability(labelIndex: Int, value: Number)

    protected abstract fun getNormalizedProbability(labelIndex: Int): Float

    protected abstract fun runInference()

    protected val numLabels: Int
        get() = labels.size
}