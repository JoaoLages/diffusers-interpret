// @ts-check

((d) => {
  /**
   * Constants
   */
  const ID_SLIDER = "slider";
  const ID_LOADING = "loading";
  const ID_ERROR = "error";
  //
  const ID_BUTTON_PREV = "slider-action-prev";
  const ID_BUTTON_NEXT = "slider-action-next";
  //
  const ID_IMAGE_FIRST = "slider-image-first";
  const ID_IMAGE_CURRENT = "slider-image-current";
  const ID_IMAGE_FINAL = "slider-image-final";
  //
  const ID_ITERATIONS_CURRENT = "slider-iterations-current";
  const ID_ITERATIONS_FINAL = "slider-iterations-final";

  /**
   * @type {{image: string}[]}
   */
  let imageList = [];

  /**
   * @type {number}
   */
  let currentIndex = 0;

  /**
   * Initialize the Image Slider
   *
   * @param {string} jsonPayload
   */
  function initialize(jsonPayload) {
    const isOK = parseJSONPayload(jsonPayload);

    if (isOK) {
      handleSuccessState();
    } else {
      handleErrorState();
    }
  }

  /**
   * Parse the JSON payload
   *
   * @param {string} jsonPayload
   *
   * @return {boolean}
   */
  function parseJSONPayload(jsonPayload) {
    if (Array.isArray(jsonPayload)) {
      imageList = jsonPayload;

      return true;
    }

    return false;
  }

  /**
   * Render the application success state
   */
  function handleSuccessState() {
    /**
     * Update First Image
     */
    const imageFirst = imageList[0] ?? {};
    updateImageAttributes(ID_IMAGE_FIRST, {
      backgroundImage: imageFirst?.image ?? null,
    });

    /**
     * Update Current Image
     */
    const imageCurrent = imageList[currentIndex] ?? {};
    updateImageAttributes(ID_IMAGE_CURRENT, {
      backgroundImage: imageCurrent?.image ?? null,
    });

    /**
     * Update Final Image
     */
    const imageFinal = imageList[imageList.length - 1] ?? {};
    updateImageAttributes(ID_IMAGE_FINAL, {
      backgroundImage: imageFinal?.image ?? null,
    });

    /**
     * Update the iteration values
     */
    updateIterationValues();

    /**
     * Set Prev Button initial state
     */
    updateButtonAttributes(ID_BUTTON_PREV, {
      disabled: currentIndex === 0,
    });

    /**
     * Set Next Button initial state
     */
    const imageLen = imageList.length;
    updateButtonAttributes(ID_BUTTON_NEXT, {
      disabled: currentIndex === imageLen - 1,
    });

    /**
     * Initialize a `click` event in the Prev Button
     */
    const $actionPrev = d.getElementById(ID_BUTTON_PREV);
    if ($actionPrev) $actionPrev.addEventListener("click", prevImageAction);

    /**
     * Initialize a `click` event in the Next Button
     */
    const $actionNext = d.getElementById(ID_BUTTON_NEXT);
    if ($actionNext) $actionNext.addEventListener("click", nextImageAction);

    hideElement(ID_LOADING);
    showElement(ID_SLIDER);
  }

  /**
   * Render the application error state
   */
  function handleErrorState() {
    hideElement(ID_LOADING);
    showElement(ID_ERROR);
  }

  /**
   * Click the `Prev` image button
   */
  function prevImageAction() {
    const canGoPrev = currentIndex > 0;

    if (canGoPrev) {
      currentIndex--;

      const backgroundImage = imageList[currentIndex]?.image ?? null;

      updateImageAttributes(ID_IMAGE_CURRENT, { backgroundImage });
      updateButtonAttributes(ID_BUTTON_NEXT, { disabled: false });
      updateIterationValues();

      const disablePrev = currentIndex === 0;

      if (disablePrev) {
        updateButtonAttributes(ID_BUTTON_PREV, { disabled: true });
      } else {
        updateButtonAttributes(ID_BUTTON_NEXT, { disabled: false });
      }
    }
  }

  /**
   * Click the `Next` image button
   */
  function nextImageAction() {
    const imageLen = imageList.length;
    const canGoNext = currentIndex < imageLen - 1;

    if (canGoNext) {
      currentIndex++;

      const backgroundImage = imageList[currentIndex]?.image ?? null;

      updateImageAttributes(ID_IMAGE_CURRENT, { backgroundImage });
      updateButtonAttributes(ID_BUTTON_PREV, { disabled: false });
      updateIterationValues();

      const disableNext = currentIndex === imageLen - 1;

      if (disableNext) {
        updateButtonAttributes(ID_BUTTON_NEXT, { disabled: true });
      } else {
        updateButtonAttributes(ID_BUTTON_PREV, { disabled: false });
      }
    }
  }

  /**
   * Update the iteration values
   */
  function updateIterationValues() {
    const $iterationCurrent = d.getElementById(ID_ITERATIONS_CURRENT);
    const $iterationFinal = d.getElementById(ID_ITERATIONS_FINAL);

    if ($iterationCurrent) {
      $iterationCurrent.innerText = `${currentIndex + 1}`;
    }

    if ($iterationFinal) {
      const len = imageList?.length ?? 0;

      $iterationFinal.innerText = `${len}`;
    }
  }

  /**
   * Show an element
   *
   * @param {ID_SLIDER | ID_LOADING | ID_ERROR} id
   */
  function showElement(id) {
    const $element = d.getElementById(id);

    if ($element) $element.style.display = "flex";
  }

  /**
   * Hide an element
   *
   * @param {ID_SLIDER | ID_LOADING | ID_ERROR} id
   */
  function hideElement(id) {
    const $element = d.getElementById(id);

    if ($element) $element.style.display = "none";
  }

  /**
   * Update the Image attributes
   *
   * @param {ID_IMAGE_FIRST | ID_IMAGE_CURRENT | ID_IMAGE_FINAL} id
   * @param {{ backgroundImage: string | null}} options
   */
  function updateImageAttributes(id, options) {
    const { backgroundImage } = options ?? {};

    if (id) {
      const $img = d.getElementById(id);

      if ($img && backgroundImage) {
        $img.style.backgroundImage = `url("${backgroundImage}")`;
      }
    }
  }

  /**
   * Update the Prev/Next Button attributes
   *
   * @param {ID_BUTTON_PREV | ID_BUTTON_NEXT} id
   * @param {{disabled: boolean}} options
   */
  function updateButtonAttributes(id, options) {
    const { disabled } = options ?? {};

    if (id) {
      const $button = d.getElementById(id);

      // @ts-ignore
      if ($button) $button.disabled = disabled;
    }
  }

  /**
   * Trigger the `INITIALIZE_IS_READY` event when the Document is ready.
   */
  d.addEventListener("DOMContentLoaded", function isReady() {
    const $body = d.querySelector("body");

    if ($body) {
      const e = new CustomEvent("INITIALIZE_IS_READY", {
        detail: { initialize },
      });

      $body.dispatchEvent(e);
    }
  });
})(document);
