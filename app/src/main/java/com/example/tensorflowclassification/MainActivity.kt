package com.example.tensorflowclassification

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.*
import android.util.Size
import android.os.Handler
import android.os.HandlerThread
import android.os.Message
import android.util.Rational
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.tensorflowclassification.tflite.Classifier
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit

// This is an arbitrary number we are using to keep tab of the permission
// request. Where an app has multiple context for requesting permission,
// this can help differentiate the different contexts
private const val REQUEST_CODE_PERMISSIONS = 10

// This is an array of all the permission specified in the manifest
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

class MainActivity : AppCompatActivity(), LifecycleOwner {

    private lateinit var viewFinder: TextureView
    private lateinit var label: TextView

    private inner class TensorflowModelAnalyzer( val activity: Activity, val labelHandler: Handler ): ImageAnalysis.Analyzer {
        private var lastAnalyzedTimestamp = 0L

        private val classifier = Classifier.create(activity, Classifier.Model.QUANTIZED, Classifier.Device.CPU, 2 )

        private fun ByteBuffer.toByteArray(): ByteArray {
            rewind()
            val data = ByteArray(remaining())
            get(data)
            return data
        }

        private fun convertToBitmap( image: ImageProxy ): Bitmap {
            val yBuffer = image.planes[0].buffer // Y
            val uBuffer = image.planes[1].buffer // U
            val vBuffer = image.planes[2].buffer // V

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            //U and V are swapped
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
            val imageBytes = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }


        override fun analyze(image: ImageProxy, rotationDegrees: Int) {
            val currentTimestamp = System.currentTimeMillis()

            if (currentTimestamp - lastAnalyzedTimestamp >= TimeUnit.SECONDS.toMillis(3)) {
                val bitmapPicture = convertToBitmap( image )

                val result = classifier.recognizeImage(bitmapPicture)[0]
                labelHandler.obtainMessage(1, result.title)?.apply {
                    sendToTarget()
                }

                // Update timestamp of last analyzed frame
                lastAnalyzedTimestamp = currentTimestamp
            }
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.view_finder)
        label = findViewById(R.id.label)

        if(allPermissionsGranted()) {
            viewFinder.post { startCamera() }
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        viewFinder.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            updateTransform()
        }
    }

    private fun startCamera(){
        val previewConfig = PreviewConfig.Builder().apply {
            setTargetAspectRatio(Rational(1, 1))
            setTargetResolution(Size(640, 480))
        }.build()

        val preview = Preview(previewConfig)

        preview.setOnPreviewOutputUpdateListener {
            // To update the SurfaceTexture, we have to remove it and re-add it
            val parent = viewFinder.parent as ViewGroup
            parent.removeView(viewFinder)
            parent.addView(viewFinder, 0)

            viewFinder.surfaceTexture = it.surfaceTexture
            updateTransform()
        }

        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            val analyzerThread = HandlerThread("ModelClassification").apply { start() }
            setCallbackHandler(Handler(analyzerThread.looper))
            setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
        }.build()

        val activity = this
        val labelHandler = object: Handler( this.mainLooper ) {
            override fun handleMessage(msg: Message?) {
                label.text = msg?.obj.toString() ?: ""
            }
        }

        val analyzerUseCase = ImageAnalysis(analyzerConfig).apply {
            analyzer = TensorflowModelAnalyzer(activity, labelHandler)
        }

        CameraX.bindToLifecycle(this, preview, analyzerUseCase)
    }

    private fun updateTransform() {
        val matrix = Matrix()

        val centerX = viewFinder.width / 2f
        val centerY = viewFinder.height / 2f

        // Correct preview output to account for display rotation
        val rotationDegress = when(viewFinder.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }

        matrix.postRotate(-rotationDegress.toFloat(), centerX, centerY)

        // Finally, apply transformations to our TextureView
        viewFinder.setTransform(matrix)
    }

    /**
     * Process result from permission request dialog box, has the request
     * been granted? If yes, start Camera. Otherwise display a toast
     */
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS){
            if(allPermissionsGranted()){
                viewFinder.post { startCamera() }
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
}
