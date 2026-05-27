<script setup lang="ts">
import { useSlideContext, useNav } from '@slidev/client'

const { $slidev } = useSlideContext()
const { currentPage, total } = useNav()
const title = $slidev?.configs?.title ?? ''
const speaker = $slidev?.configs?.author ?? ''
</script>

<template>
  <div class="slidev-layout epfl-content" style="padding: 36px 90px 70px 90px;">
    <div class="epfl-chrome-rule" />
    <div class="epfl-chrome-top">
      <span v-if="title" class="epfl-chrome-title">{{ title }}</span>
      <img src="/epfl/logo.svg" class="epfl-chrome-logo" alt="EPFL" />
    </div>

    <div class="two-cols-header"><slot /></div>

    <div class="two-cols-grid">
      <div class="col-left"><slot name="left" /></div>
      <div class="col-right"><slot name="right" /></div>
    </div>

    <div class="two-cols-footer"><slot name="footer" /></div>

    <div class="epfl-chrome-footer">
      <div></div>
      <div>{{ currentPage }} / {{ total }}</div>
    </div>
  </div>
</template>

<style scoped>
.two-cols-header { margin-bottom: 4px; }
.two-cols-header :empty { display: none; }

.two-cols-footer { margin-top: 16px; }
.two-cols-footer:empty { display: none; }

.two-cols-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  align-items: start;
}
</style>
