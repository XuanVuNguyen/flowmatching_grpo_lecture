<script setup lang="ts">
import { useSlideContext, useNav } from '@slidev/client'

const { $slidev } = useSlideContext()
const { currentPage, total } = useNav()
const title = $slidev?.configs?.title ?? ''
const speaker = $slidev?.configs?.author ?? ''
</script>

<template>
  <div class="slidev-layout epfl-content" style="padding: 36px 60px 52px 60px;">
    <div class="epfl-chrome-rule" />
    <div class="epfl-chrome-top">
      <span v-if="title" class="epfl-chrome-title">{{ title }}</span>
      <img src="/epfl/logo.svg" class="epfl-chrome-logo" alt="EPFL" />
    </div>

    <div class="epfl-content-body">
      <slot />
    </div>

    <div class="epfl-chrome-footer">
      <div></div>
      <div>{{ currentPage }} / {{ total }}</div>
    </div>
  </div>
</template>
